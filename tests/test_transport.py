import asyncio

import httpx
import pytest
from conftest import FragmentStream, events_bytes

from subscription_gateway.errors import GatewayError
from subscription_gateway.providers.responses import decode_sse


@pytest.mark.parametrize("fragment", [1, 2, 7, 4096])
async def test_fragmented_utf8_and_multiline_data(fragment):
    raw = (
        'data: {"type": "response.output_text.delta",\r\ndata: "delta": "🌍 café"}\r\n\r\n'
    ).encode()
    stream = FragmentStream(raw, fragment)
    response = httpx.Response(200, stream=stream)
    assert [event async for event in decode_sse(response)] == [
        {"type": "response.output_text.delta", "delta": "🌍 café"}
    ]
    await response.aclose()
    assert stream.closed


@pytest.mark.parametrize(
    "raw,code",
    [
        (b'data: {"type":"x"}\n', "upstream_interrupted"),
        (b"data: not-json\n\n", "upstream_protocol"),
        (b"data: \xff\n\n", "upstream_protocol"),
        (b"data: " + b"x" * 256, "upstream_protocol"),
    ],
)
async def test_invalid_sse_and_limits(raw, code):
    response = httpx.Response(200, stream=FragmentStream(raw, 3))
    with pytest.raises(GatewayError) as error:
        _ = [event async for event in decode_sse(response, max_event_bytes=128)]
    assert error.value.code == code
    await response.aclose()


async def test_provider_cancellation_closes_transport(fake_gateway):
    _, _, provider = fake_gateway
    started, closed = asyncio.Event(), asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield events_bytes([{"type": "response.created"}])
            started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async def handler(_request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=SlowStream()
        )

    await provider.client.aclose()
    provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    iterator = provider.events({"model": "test-model"})
    await anext(iterator)
    task = asyncio.create_task(anext(iterator))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    await provider.client.aclose()


@pytest.mark.parametrize("separator", [b"\n", b"\r", b"\r\n"])
async def test_all_sse_line_endings(separator):
    response = httpx.Response(
        200, stream=FragmentStream(b'data: {"type":"response.created"}' + separator + separator, 1)
    )
    assert [event async for event in decode_sse(response)] == [{"type": "response.created"}]
    await response.aclose()
