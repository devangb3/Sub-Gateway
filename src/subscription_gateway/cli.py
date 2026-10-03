import argparse
import asyncio
import logging
import os
from pathlib import Path

import httpx
import uvicorn

from subscription_gateway.auth.oauth import OAuthManager
from subscription_gateway.auth.store import FileCredentialStore
from subscription_gateway.errors import GatewayError
from subscription_gateway.http.app import create_app
from subscription_gateway.providers.responses import ResponsesProvider


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Local ChatGPT plan Chat Completions gateway")
    result.add_argument(
        "--credentials-dir", type=Path, default=Path.home() / ".subscription-gateway"
    )
    commands = result.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login", help="Continue with ChatGPT")
    login.add_argument("--callback-port", type=int, default=1455)
    commands.add_parser("logout")
    commands.add_parser("status")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "::1"])
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--cors-origin", action="append", default=[])
    serve.add_argument("--concurrency", type=int, default=4)
    serve.add_argument("--timeout", type=float, default=180)
    serve.add_argument("--max-body-bytes", type=int, default=1_048_576)
    return result


async def run(args) -> None:
    store = FileCredentialStore(args.credentials_dir)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30, read=60), follow_redirects=False, trust_env=False
    ) as client:
        oauth = OAuthManager(store, client)
        if args.command == "login":
            await oauth.login(args.callback_port)
            print("Signed in. ChatGPT plan usage authorized.")
        elif args.command == "logout":
            confirmed = await oauth.logout()
            print("Signed out locally.")
            if not confirmed:
                print(
                    "Remote revocation was not confirmed. Disconnect the app in ChatGPT Settings."
                )
        elif args.command == "status":
            async with store.locked():
                credentials = store.read()
            active = bool(credentials and credentials.access_token)
            print(
                "Signed in (local credentials; account access not probed)."
                if active
                else "Signed out. Run subscription-gateway login."
            )
        elif args.command == "serve":
            key = os.environ.get("SUBSCRIPTION_GATEWAY_KEY", "")
            provider = ResponsesProvider(oauth, client)
            app = create_app(
                provider,
                key,
                cors_origins=tuple(args.cors_origin),
                concurrency=args.concurrency,
                request_timeout=args.timeout,
                max_body_bytes=args.max_body_bytes,
            )
            # Access logs include caller-controlled URLs; leave them disabled.
            config = uvicorn.Config(
                app, host=args.host, port=args.port, access_log=False, timeout_graceful_shutdown=10
            )
            await uvicorn.Server(config).serve()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(run(parser().parse_args()))
    except (GatewayError, ValueError, OSError) as error:
        message = (
            error.message
            if isinstance(error, GatewayError)
            else "Invalid configuration or local I/O failure."
        )
        raise SystemExit(message) from None
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
