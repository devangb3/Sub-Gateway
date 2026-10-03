import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx

from subscription_gateway.auth.oauth import OAuthManager
from subscription_gateway.errors import GatewayError
from subscription_gateway.providers.base import ProviderEvent


def provider_error(status: int, code: str | None = None) -> GatewayError:
    if code in (
        "subscription_sharing_usage_limit_exceeded",
        "subscription_sharing_usage_unavailable",
    ):
        return GatewayError(
            429, code, "ChatGPT plan usage is unavailable or its limit was reached."
        )
    if status in (401, 403):
        return GatewayError(401, "reauthentication_required", "Run subscription-gateway login.")
    if status == 429:
        return GatewayError(429, "subscription_limit", "ChatGPT plan request limit reached.")
    if status in (400, 404):
        return GatewayError(400, "upstream_rejected", "Upstream rejected the model or request.")
    return GatewayError(502, "upstream_unavailable", "Inference service unavailable.")


async def decode_sse(
    response: httpx.Response, max_event_bytes: int = 2_097_152
) -> AsyncIterator[ProviderEvent]:
    buffer = bytearray()
    data: list[str] = []
    size = 0
    skip_lf = False
    async for fragment in response.aiter_bytes():
        buffer.extend(fragment)
        if skip_lf and buffer:
            if buffer[0] == 10:
                del buffer[:1]
            skip_lf = False
        while True:
            cr, lf = buffer.find(b"\r"), buffer.find(b"\n")
            positions = [position for position in (cr, lf) if position >= 0]
            if not positions:
                break
            position = min(positions)
            # CR ends a line immediately; swallow a following LF across fragments.
            skip_lf = buffer[position] == 13 and position == len(buffer) - 1
            if position > max_event_bytes:
                raise GatewayError(502, "upstream_protocol", "Upstream event exceeds limit.")
            raw_line = bytes(buffer[:position])
            separator = 2 if buffer[position : position + 2] == b"\r\n" else 1
            del buffer[: position + separator]
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise GatewayError(502, "upstream_protocol", "Invalid upstream UTF-8.") from exc
            if not line:
                if data:
                    raw = "\n".join(data)
                    if raw != "[DONE]":
                        try:
                            event = json.loads(raw)
                            if not isinstance(event, dict) or not isinstance(
                                event.get("type"), str
                            ):
                                raise ValueError()
                        except ValueError as exc:
                            raise GatewayError(
                                502, "upstream_protocol", "Invalid upstream event."
                            ) from exc
                        yield event
                data, size = [], 0
            elif line.startswith("data:"):
                part = line[5:].removeprefix(" ")
                size += len(raw_line)
                if size > max_event_bytes:
                    raise GatewayError(502, "upstream_protocol", "Upstream event exceeds limit.")
                data.append(part)
        if len(buffer) > max_event_bytes:
            raise GatewayError(502, "upstream_protocol", "Upstream event exceeds limit.")
    if data or buffer:
        raise GatewayError(502, "upstream_interrupted", "Upstream ended inside an SSE event.")


class ResponsesProvider:
    def __init__(
        self,
        oauth: OAuthManager,
        client: httpx.AsyncClient,
        base_url: str = "https://api.openai.com/v1",
        cache_seconds: float = 60,
    ):
        self.oauth = oauth
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.cache_seconds = cache_seconds
        self._models: list[dict] = []
        self._cache_until = 0.0
        self._cache_token: str | None = None
        self._model_lock = asyncio.Lock()

    async def models(self) -> list[dict]:
        token = await self.oauth.access_token()
        async with self._model_lock:
            if time.monotonic() < self._cache_until and self._cache_token == token:
                return self._models
            try:
                response = await self.client.get(
                    self.base_url + "/models", headers={"Authorization": "Bearer " + token}
                )
                if response.status_code != 200:
                    raise provider_error(response.status_code)
                models = response.json()["models"]
                if not isinstance(models, list):
                    raise ValueError()
                self._models = [
                    {
                        "id": "openai/" + model["slug"],
                        "name": model.get("display_name", model["slug"]),
                        "object": "model",
                        "owned_by": "openai",
                    }
                    for model in models
                    if model.get("visibility") == "list"
                ]
                self._cache_token = token
                self._cache_until = time.monotonic() + self.cache_seconds
                return self._models
            except httpx.TimeoutException as exc:
                raise GatewayError(504, "upstream_timeout", "Model discovery timed out.") from exc
            except httpx.HTTPError as exc:
                raise provider_error(502) from exc
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise GatewayError(
                    502, "upstream_protocol", "Invalid upstream model catalog."
                ) from exc

    async def events(self, request: dict) -> AsyncIterator[ProviderEvent]:
        token = await self.oauth.access_token()
        try:
            async with self.client.stream(
                "POST",
                self.base_url + "/responses",
                json=request,
                headers={"Authorization": "Bearer " + token},
            ) as response:
                if response.status_code != 200:
                    # Do not reflect raw upstream response bodies.
                    raise provider_error(response.status_code)
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise GatewayError(
                        502, "upstream_protocol", "Expected an upstream event stream."
                    )
                async for event in decode_sse(response):
                    yield event
        except httpx.TimeoutException as exc:
            raise GatewayError(504, "upstream_timeout", "Inference timed out.") from exc
        except httpx.HTTPError as exc:
            raise provider_error(502) from exc

    async def aclose(self) -> None:
        await self.client.aclose()
