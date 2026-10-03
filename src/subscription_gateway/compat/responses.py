import json
import time
import uuid

from subscription_gateway.errors import GatewayError
from subscription_gateway.providers.responses import provider_error


def usage_from_response(usage: dict | None) -> dict | None:
    if usage is None:
        return None
    prompt = usage["input_tokens"]
    completion = usage["output_tokens"]
    if any(type(value) is not int or value < 0 for value in (prompt, completion)):
        raise ValueError("Invalid usage")
    result = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": usage.get("total_tokens", prompt + completion),
    }
    cached = usage.get("input_tokens_details", {}).get("cached_tokens")
    reasoning = usage.get("output_tokens_details", {}).get("reasoning_tokens")
    if cached is not None:
        result["prompt_tokens_details"] = {"cached_tokens": cached}
    if reasoning is not None:
        result["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    counts = [result["total_tokens"], cached, reasoning]
    if any(value is not None and (type(value) is not int or value < 0) for value in counts):
        raise ValueError("Invalid usage details")
    return result


class Normalizer:
    def __init__(self, model: str, output_limit: int = 16_777_216):
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.created = int(time.time())
        self.model = model
        self.text: dict[tuple[int, int], str] = {}
        self.calls: dict[int, dict] = {}
        self.reasoning: list[dict] = []
        self.completed = False
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.output_limit = output_limit
        self.output_size = 0

    def chunk(self, delta: dict, finish: str | None = None) -> dict:
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "provider": "openai",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def error_chunk(self, error: GatewayError) -> dict:
        result = self.chunk({"content": ""}, "error")
        result.update(error.envelope())
        return result

    def account(self, value: str) -> None:
        self.output_size += len(value.encode())
        if self.output_size > self.output_limit:
            raise GatewayError(502, "output_limit", "Upstream output exceeded gateway limit.")

    def add_call(self, output_index: int, item: dict) -> list[dict]:
        if output_index in self.calls:
            call = self.calls[output_index]
            if call["id"] != item["call_id"] or call["function"]["name"] != item["name"]:
                raise GatewayError(
                    502, "upstream_protocol", "Function identity changed during stream."
                )
            return []
        if any(call["id"] == item["call_id"] for call in self.calls.values()):
            raise GatewayError(502, "upstream_protocol", "Duplicate upstream function call ID.")
        call = {
            "index": len(self.calls),
            "id": item["call_id"],
            "type": "function",
            "function": {"name": item["name"], "arguments": ""},
        }
        self.calls[output_index] = call
        self.account(call["id"] + call["function"]["name"])
        return [self.chunk({"tool_calls": [{**call, "function": dict(call["function"])}]})]

    def arguments(self, index: int, value: str, final: bool = False) -> list[dict]:
        call = self.calls[index]
        previous = call["function"]["arguments"]
        if final:
            if not value.startswith(previous):
                raise GatewayError(
                    502, "upstream_protocol", "Function arguments changed during stream."
                )
            value = value[len(previous) :]
        if not value:
            return []
        self.account(value)
        call["function"]["arguments"] += value
        return [
            self.chunk({"tool_calls": [{"index": call["index"], "function": {"arguments": value}}]})
        ]

    def add_text(self, key: tuple[int, int], value: str, final: bool = False) -> list[dict]:
        previous = self.text.get(key, "")
        if final:
            if not value.startswith(previous):
                raise GatewayError(502, "upstream_protocol", "Text changed during stream.")
            value = value[len(previous) :]
        if not value:
            return []
        self.account(value)
        self.text[key] = previous + value
        return [self.chunk({"content": value})]

    def item_done(self, index: int, item: dict) -> list[dict]:
        chunks: list[dict] = []
        if item["type"] == "function_call":
            chunks += self.add_call(index, item)
            chunks += self.arguments(index, item["arguments"], final=True)
        elif item["type"] == "reasoning" and item.get("encrypted_content"):
            detail = {
                "type": "reasoning.encrypted",
                "format": "openai-responses-v1",
                "id": item["id"],
                "data": item["encrypted_content"],
                "index": index,
            }
            if not any(old["id"] == detail["id"] for old in self.reasoning):
                self.account(detail["data"])
                self.reasoning.append(detail)
                chunks.append(self.chunk({"reasoning_details": [detail]}))
        elif item["type"] == "message":
            for content_index, content in enumerate(item.get("content", [])):
                if content["type"] == "output_text":
                    chunks += self.add_text((index, content_index), content["text"], final=True)
                elif content["type"] == "refusal":
                    # Expose refusal as text so text-only clients can display it.
                    chunks += self.add_text((index, content_index), content["refusal"], final=True)
        return chunks

    def ingest(self, event: dict) -> list[dict]:
        try:
            return self._ingest(event)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise GatewayError(
                502, "upstream_protocol", "Invalid upstream response event."
            ) from exc

    def _ingest(self, event: dict) -> list[dict]:
        kind = event["type"]
        if kind in ("response.failed", "error"):
            error = event.get("response", {}).get("error") or event.get("error") or event
            raise provider_error(502, error.get("code"))
        if kind == "response.incomplete":
            raise GatewayError(502, "upstream_incomplete", "Upstream did not complete inference.")
        if kind == "response.output_text.delta":
            return self.add_text((event["output_index"], event["content_index"]), event["delta"])
        if kind == "response.refusal.delta":
            return self.add_text((event["output_index"], event["content_index"]), event["delta"])
        if kind == "response.output_item.added" and event["item"]["type"] == "function_call":
            chunks = self.add_call(event["output_index"], event["item"])
            chunks += self.arguments(event["output_index"], event["item"].get("arguments", ""))
            return chunks
        if kind == "response.function_call_arguments.delta":
            return self.arguments(event["output_index"], event["delta"])
        if kind == "response.output_item.done":
            return self.item_done(event["output_index"], event["item"])
        if kind == "response.completed":
            response = event["response"]
            if response.get("status", "completed") != "completed":
                raise GatewayError(
                    502, "upstream_incomplete", "Upstream did not complete inference."
                )
            chunks = []
            for index, item in enumerate(response.get("output", [])):
                chunks += self.item_done(index, item)
            self.usage = usage_from_response(response.get("usage"))
            self.completed = True
            self.finish_reason = "tool_calls" if self.calls else "stop"
            for call in self.calls.values():
                if not isinstance(json.loads(call["function"]["arguments"]), dict):
                    raise ValueError("Invalid function arguments")
            terminal = self.chunk({"role": "assistant", "content": ""}, self.finish_reason)
            terminal["choices"][0]["native_finish_reason"] = self.finish_reason
            chunks.append(terminal)
            usage_chunk = self.chunk({"role": "assistant", "content": ""}, self.finish_reason)
            usage_chunk["choices"][0]["native_finish_reason"] = self.finish_reason
            usage_chunk["usage"] = self.usage
            chunks.append(usage_chunk)
            return chunks
        return []

    def require_completed(self) -> None:
        if not self.completed:
            raise GatewayError(
                502, "upstream_interrupted", "Stream ended without response.completed."
            )

    def completion(self) -> dict:
        self.require_completed()
        message: dict = {"role": "assistant", "content": "".join(self.text.values()) or None}
        if self.calls:
            message["tool_calls"] = [
                {key: value for key, value in call.items() if key != "index"}
                for call in self.calls.values()
            ]
        if self.reasoning:
            message["reasoning_details"] = self.reasoning
        result = {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "provider": "openai",
            "choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}],
        }
        if self.usage is not None:
            result["usage"] = self.usage
        return result


def encode_sse(value: dict | str) -> bytes:
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ("data: " + data + "\n\n").encode()
