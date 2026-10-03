import json
import time

import httpx
import pytest

from subscription_gateway.auth.oauth import OAuthManager
from subscription_gateway.auth.store import Credentials, FileCredentialStore
from subscription_gateway.http.app import create_app
from subscription_gateway.providers.responses import ResponsesProvider

KEY = "test-gateway-key-at-least-24-characters"
MODEL = "openai/test-model"


class FragmentStream(httpx.AsyncByteStream):
    def __init__(self, payload, fragment=7):
        self.payload = payload
        self.fragment = fragment
        self.closed = False

    async def __aiter__(self):
        for index in range(0, len(self.payload), self.fragment):
            yield self.payload[index : index + self.fragment]

    async def aclose(self):
        self.closed = True


def events_bytes(events):
    return b": keepalive\r\n\r\n" + b"".join(
        (
            "event: "
            + event["type"]
            + "\r\ndata: "
            + json.dumps(event, ensure_ascii=False)
            + "\r\n\r\n"
        ).encode()
        for event in events
    )


def text_events(text="Hello 🌍"):
    return [
        {"type": "response.created", "response": {"id": "resp-upstream"}},
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": text[:3],
        },
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": text[3:],
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "total_tokens": 30,
                    "input_tokens_details": {"cached_tokens": 8},
                    "output_tokens_details": {"reasoning_tokens": 6},
                },
            },
        },
    ]


@pytest.fixture
async def fake_gateway(tmp_path):
    store = FileCredentialStore(tmp_path / "auth")
    async with store.locked():
        store.write(
            Credentials(
                ext_agent_host_id="urn:uuid:host",
                client_id="issued-client",
                subject="user",
                access_token="private-access",
                refresh_token="private-refresh",
                expires_at=time.time() + 3600,
                scopes=["resource.invoke", "chatgpt.tokens.use.direct"],
            )
        )
    state = {
        "events": text_events(),
        "requests": [],
        "streams": [],
        "status": 200,
        "catalog_calls": 0,
    }

    async def handler(request):
        assert request.headers["authorization"] == "Bearer private-access"
        if request.url.path == "/v1/models":
            state["catalog_calls"] += 1
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"slug": "test-model", "display_name": "Test model", "visibility": "list"},
                        {"slug": "hidden-model", "visibility": "hidden"},
                    ]
                },
            )
        assert request.url.path == "/v1/responses"
        state["requests"].append(json.loads(request.content))
        if state["status"] != 200:
            return httpx.Response(state["status"], text="SECRET upstream body")
        stream = FragmentStream(events_bytes(state["events"]), fragment=1)
        state["streams"].append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = ResponsesProvider(OAuthManager(store, upstream), upstream)
    app = create_app(provider, KEY)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway",
        headers={"Authorization": "Bearer " + KEY},
    ) as client:
        yield client, state, provider
    await upstream.aclose()


def parse_chunks(text):
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
