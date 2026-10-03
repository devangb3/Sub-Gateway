import asyncio
import hmac
import json
import logging
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from typing import TypeVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.middleware.cors import CORSMiddleware

from subscription_gateway.compat.requests import translate
from subscription_gateway.compat.responses import Normalizer, encode_sse
from subscription_gateway.compat.schemas import ChatRequest
from subscription_gateway.errors import GatewayError
from subscription_gateway.providers.base import Provider

logger = logging.getLogger("subscription_gateway")
T = TypeVar("T")


async def while_connected(request: Request, operation: Awaitable[T]) -> T:
    """Cancel model/inference work when a nonstreaming client disconnects."""
    task = asyncio.ensure_future(operation)
    try:
        while not task.done():
            if await request.is_disconnected():
                raise asyncio.CancelledError()
            await asyncio.wait({task}, timeout=0.1)
        return task.result()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def create_app(
    provider: Provider,
    bearer_key: str,
    *,
    cors_origins: tuple[str, ...] = (),
    max_body_bytes: int = 1_048_576,
    concurrency: int = 4,
    request_timeout: float = 180,
) -> FastAPI:
    if not bearer_key or len(bearer_key) < 24:
        raise ValueError("Gateway bearer key must contain at least 24 characters")
    if concurrency < 1 or max_body_bytes < 1 or request_timeout <= 0:
        raise ValueError("Operational limits must be positive")
    if "*" in cors_origins:
        raise ValueError("CORS requires explicit origins")
    slots = asyncio.Semaphore(concurrency)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await provider.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.exception_handler(GatewayError)
    async def gateway_error(_request, error):
        logger.warning("request failed status=%s code=%s", error.status, error.code)
        return JSONResponse(error.envelope(), status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def schema_error(_request, _error):
        return JSONResponse(
            {"error": {"code": "invalid_request", "message": "Invalid request."}}, status_code=400
        )

    def authorize(request: Request) -> None:
        expected = "Bearer " + bearer_key
        actual = request.headers.get("authorization", "")
        if not hmac.compare_digest(actual.encode(), expected.encode()):
            raise GatewayError(
                401, "invalid_gateway_key", "A valid gateway bearer key is required."
            )

    async def read_body(request: Request) -> ChatRequest:
        body = bytearray()
        try:
            async with asyncio.timeout(min(30, request_timeout)):
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > max_body_bytes:
                        raise GatewayError(413, "request_too_large", "Request body exceeds limit.")
        except TimeoutError as exc:
            raise GatewayError(408, "request_timeout", "Request body timed out.") from exc
        try:
            return ChatRequest.model_validate(json.loads(body))
        except (ValueError, ValidationError, UnicodeDecodeError) as exc:
            # Paths identify unsupported fields without echoing submitted secrets/content.
            fields = []
            if isinstance(exc, ValidationError):
                fields = [
                    ".".join(str(part) for part in error["loc"])
                    for error in exc.errors(include_input=False)
                ]
            message = "Invalid or unsupported request fields"
            if fields:
                message += ": " + ", ".join(fields[:8])
            raise GatewayError(400, "invalid_request", message + ".") from exc

    async def acquire() -> None:
        try:
            async with asyncio.timeout(0.05):
                await slots.acquire()
        except TimeoutError as exc:
            raise GatewayError(429, "gateway_busy", "Gateway concurrency limit reached.") from exc

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request):
        authorize(request)
        await acquire()
        try:
            async with asyncio.timeout(request_timeout):
                return {"object": "list", "data": await while_connected(request, provider.models())}
        except TimeoutError as exc:
            raise GatewayError(504, "upstream_timeout", "Model discovery timed out.") from exc
        finally:
            slots.release()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        authorize(request)
        chat = await read_body(request)
        await acquire()
        deadline = asyncio.get_running_loop().time() + request_timeout
        iterator = None
        normalizer = Normalizer(chat.model)
        handed_off = False
        try:
            async with asyncio.timeout_at(deadline):
                catalog = await while_connected(request, provider.models())
                if chat.model not in {model["id"] for model in catalog}:
                    raise GatewayError(
                        400, "invalid_model", "Model is not available to this account."
                    )
                iterator = provider.events(translate(chat))
                try:
                    first = await while_connected(request, anext(iterator))
                except StopAsyncIteration as exc:
                    raise GatewayError(
                        502, "upstream_interrupted", "Empty upstream stream."
                    ) from exc
                first_chunks = normalizer.ingest(first)
                if not chat.stream:

                    async def aggregate():
                        if not normalizer.completed:
                            async for event in iterator:
                                normalizer.ingest(event)
                                if normalizer.completed:
                                    break
                        return normalizer.completion()

                    return await while_connected(request, aggregate())

            async def stream():
                try:
                    async with asyncio.timeout_at(deadline):
                        yield encode_sse(normalizer.chunk({"role": "assistant", "content": ""}))
                        for chunk in first_chunks:
                            yield encode_sse(chunk)
                        if not normalizer.completed:
                            async for event in iterator:
                                for chunk in normalizer.ingest(event):
                                    yield encode_sse(chunk)
                                if normalizer.completed:
                                    break
                        normalizer.require_completed()
                        yield encode_sse("[DONE]")
                except TimeoutError:
                    yield encode_sse(
                        normalizer.error_chunk(
                            GatewayError(504, "upstream_timeout", "Inference timed out.")
                        )
                    )
                except GatewayError as error:
                    logger.warning("stream failed code=%s", error.code)
                    yield encode_sse(normalizer.error_chunk(error))
                except Exception:
                    logger.error("stream failed code=internal_error")
                    yield encode_sse(
                        normalizer.error_chunk(
                            GatewayError(502, "internal_error", "Gateway stream failed.")
                        )
                    )

            class OwnedStreamingResponse(StreamingResponse):
                async def __call__(self, scope, receive, send):
                    # A disconnect may occur while send() is blocked, with the
                    # generator suspended at yield, or even before its first pull.
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        try:
                            await self.body_iterator.aclose()
                        finally:
                            try:
                                await iterator.aclose()
                            finally:
                                slots.release()

            response = OwnedStreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Generation-Id": normalizer.id,
                },
            )
            handed_off = True
            return response
        except TimeoutError as exc:
            raise GatewayError(504, "upstream_timeout", "Inference timed out.") from exc
        finally:
            if not handed_off:
                try:
                    if iterator is not None:
                        await iterator.aclose()
                finally:
                    slots.release()

    return app
