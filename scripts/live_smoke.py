"""Explicitly opt-in account acceptance check. No tool implementation is executed."""

import argparse
import os

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    client = OpenAI(
        api_key=os.environ["SUBSCRIPTION_GATEWAY_KEY"],
        base_url="http://127.0.0.1:8787/v1",
        max_retries=0,
        timeout=180,
    )
    try:
        health = client.get("http://127.0.0.1:8787/health", cast_to=dict)
        assert health["status"] == "ok"
        models = client.models.list()
        assert args.model in {model.id for model in models.data}
        messages = [{"role": "user", "content": "Say hello briefly."}]
        result = client.chat.completions.create(model=args.model, messages=messages)
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message.content
        print("Health, account models and JSON: passed")
        chunks = list(
            client.chat.completions.create(model=args.model, messages=messages, stream=True)
        )
        assert chunks[-1].choices[0].finish_reason == "stop"
        assert chunks[-1].usage is not None
        assert any(chunk.choices[0].delta.content for chunk in chunks)
        print("SSE and provider usage: passed")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        history = [{"role": "user", "content": "Use lookup_weather for Toronto and summarize."}]
        call_response = client.chat.completions.create(
            model=args.model, messages=history, tools=tools, tool_choice="required"
        )
        assistant = call_response.choices[0].message
        assert assistant.tool_calls
        replay = {
            "role": "assistant",
            "content": assistant.content,
            "tool_calls": [call.model_dump(exclude_none=True) for call in assistant.tool_calls],
        }
        details = (assistant.model_extra or {}).get("reasoning_details")
        if details:
            replay["reasoning_details"] = details
        history.append(replay)
        history.extend(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": '{"city":"Toronto","weather":"sunny","source":"smoke fixture"}',
            }
            for call in assistant.tool_calls
        )
        final = client.chat.completions.create(
            model=args.model, messages=history, tools=tools, tool_choice="none"
        )
        assert final.choices[0].finish_reason == "stop"
        assert final.choices[0].message.content
        print("Function call and supplied tool-result round trip: passed")
    finally:
        client.close()


if __name__ == "__main__":
    main()
