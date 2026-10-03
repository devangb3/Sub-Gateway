import asyncio
import json

import httpx
import pytest
from conftest import KEY, MODEL, parse_chunks

from subscription_gateway.http.app import create_app


class BlockingProvider:
    def __init__(self, first=True):
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.first = first
        self.pulls = 0

    async def models(self):
        return [{"id": MODEL}]

    async def events(self, _request):
        try:
            if self.first:
                self.pulls += 1
                yield {"type": "response.created"}
            self.started.set()
            await asyncio.Event().wait()
        finally:
            self.closed.set()

    async def aclose(self):
        self.closed.set()


def request_payload(stream=False):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "stream": stream}


@pytest.mark.parametrize("stream,first", [(False, True), (True, False), (True, True)])
async def test_timeout_before_and_after_headers(stream, first):
    provider = BlockingProvider(first)
    app = create_app(provider, KEY, request_timeout=0.05)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://local",
        headers={"Authorization": "Bearer " + KEY},
    ) as client:
        response = await client.post("/v1/chat/completions", json=request_payload(stream))
    if stream and first:
        assert response.status_code == 200
        assert parse_chunks(response.text)[-1]["error"]["code"] == "upstream_timeout"
        assert "[DONE]" not in response.text
    else:
        assert response.status_code == 504
    assert provider.closed.is_set()


async def test_concurrency_bound_and_slot_release():
    provider = BlockingProvider()
    app = create_app(provider, KEY, concurrency=1, request_timeout=5)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://local",
        headers={"Authorization": "Bearer " + KEY},
    ) as client:
        task = asyncio.create_task(client.post("/v1/chat/completions", json=request_payload()))
        await asyncio.wait_for(provider.started.wait(), 1)
        overloaded = await client.post("/v1/chat/completions", json=request_payload())
        assert overloaded.status_code == 429
        assert overloaded.json()["error"]["code"] == "gateway_busy"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.closed.is_set()
        assert (await client.get("/v1/models")).status_code == 200


async def test_body_limit_and_cors():
    app = create_app(
        BlockingProvider(), KEY, max_body_bytes=32, cors_origins=("http://approved.local",)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://local",
        headers={"Authorization": "Bearer " + KEY},
    ) as client:
        assert (
            await client.post("/v1/chat/completions", json=request_payload())
        ).status_code == 413
        accepted = await client.options(
            "/v1/chat/completions",
            headers={"Origin": "http://approved.local", "Access-Control-Request-Method": "POST"},
        )
        assert accepted.headers["access-control-allow-origin"] == "http://approved.local"
        rejected = await client.options(
            "/v1/chat/completions",
            headers={"Origin": "http://unapproved.local", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in rejected.headers


@pytest.mark.parametrize("stream", [False, True])
async def test_disconnect_cancels_upstream_via_asgi(stream):
    provider = BlockingProvider()
    app = create_app(provider, KEY, request_timeout=2)
    disconnected = asyncio.Event()
    received_body = False
    frames = []
    body = json.dumps(request_payload(stream)).encode()

    async def receive():
        nonlocal received_body
        if not received_body:
            received_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        frames.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [
            (b"authorization", ("Bearer " + KEY).encode()),
            (b"content-type", b"application/json"),
        ],
        "server": ("127.0.0.1", 8787),
        "client": ("127.0.0.1", 12345),
    }
    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(provider.started.wait(), 1)
    disconnected.set()
    try:
        await asyncio.wait_for(task, 1)
    except asyncio.CancelledError:
        pass
    assert provider.closed.is_set()
    assert not any(b"[DONE]" in frame.get("body", b"") for frame in frames)


async def test_sse_backpressure_stops_pulling():
    provider = BlockingProvider()
    app = create_app(provider, KEY)
    body = json.dumps(request_payload(True)).encode()
    delivered = False
    sending = asyncio.Event()

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.body":
            sending.set()
            await asyncio.Event().wait()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"authorization", ("Bearer " + KEY).encode())],
    }
    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(sending.wait(), 1)
    assert provider.pulls == 1
    assert not provider.started.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Closing the downstream generator while suspended at yield must close upstream.
    await asyncio.sleep(0)
    assert provider.closed.is_set()
