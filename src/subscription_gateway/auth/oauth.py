import asyncio
import base64
import hashlib
import hmac
import secrets
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import jwt

from subscription_gateway.auth.store import Credentials, CredentialStore
from subscription_gateway.errors import GatewayError

ISSUER = "https://auth.openai.com"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
REQUIRED_SCOPES = {"resource.invoke", "chatgpt.tokens.use.direct"}


@dataclass
class LoginAttempt:
    redirect_uri: str
    client_id: str | None = None
    state: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    verifier: str = field(default_factory=lambda: secrets.token_urlsafe(64))
    expires_at: float = field(default_factory=lambda: time.monotonic() + 600)
    consumed: bool = False

    def url(self, credentials: Credentials) -> str:
        challenge = base64.urlsafe_b64encode(hashlib.sha256(self.verifier.encode()).digest())
        params = {
            "client_id": self.client_id or "dynamic_agent_client",
            "ext_agent_host_id": credentials.ext_agent_host_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": self.state,
            "nonce": self.nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge.decode().rstrip("="),
        }
        if self.client_id is None:
            params["agent_name_hint"] = "Subscription Gateway"
        elif credentials.id_token:
            params["id_token_hint"] = credentials.id_token
        return AUTHORIZE + "?" + urlencode(params)

    def callback(self, params: dict[str, str]) -> tuple[str, str]:
        if self.consumed or time.monotonic() >= self.expires_at:
            raise GatewayError(400, "oauth_expired", "Sign-in attempt expired or already used.")
        if not hmac.compare_digest(params.get("state", ""), self.state):
            raise GatewayError(400, "oauth_state", "Invalid OAuth state.")
        self.consumed = True
        if "error" in params:
            raise GatewayError(403, "oauth_denied", "Sign-in permission was declined.")
        client_id = params.get("client_id", self.client_id)
        if not client_id or client_id == "dynamic_agent_client":
            raise GatewayError(400, "oauth_registration", "Registration did not issue a client ID.")
        if self.client_id and client_id != self.client_id:
            raise GatewayError(
                400, "oauth_registration", "Sign-in returned a different registration."
            )
        if not params.get("code"):
            raise GatewayError(400, "oauth_code", "Sign-in did not return a code.")
        return params["code"], client_id


class OAuthManager:
    def __init__(self, store: CredentialStore, client: httpx.AsyncClient):
        self.store = store
        self.client = client

    async def post_token(self, fields: dict) -> dict:
        try:
            response = await self.client.post(TOKEN, data={**fields, "resource": RESOURCE})
            if response.status_code in (400, 401, 403):
                raise GatewayError(
                    401, "reauthentication_required", "Run subscription-gateway login."
                )
            if response.status_code != 200:
                raise GatewayError(503, "oauth_unavailable", "Authentication service unavailable.")
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Invalid token response")
            return payload
        except (httpx.HTTPError, ValueError) as exc:
            raise GatewayError(
                503, "oauth_unavailable", "Authentication service unavailable."
            ) from exc

    @staticmethod
    def merge_tokens(old: Credentials, tokens: dict) -> Credentials:
        scope = tokens.get("scope", " ".join(old.scopes))
        if not isinstance(scope, str):
            raise GatewayError(502, "oauth_invalid_response", "Invalid granted scopes.")
        scopes = scope.split()
        if not REQUIRED_SCOPES.issubset(scopes):
            raise GatewayError(
                403, "plan_permission_required", "Authorize ChatGPT plan usage at login."
            )
        access = tokens.get("access_token")
        expires = tokens.get("expires_in")
        if (
            not isinstance(access, str)
            or not access
            or isinstance(expires, bool)
            or not isinstance(expires, (int, float))
            or expires <= 0
            or not isinstance(tokens.get("token_type", "Bearer"), str)
            or tokens.get("token_type", "Bearer").lower() != "bearer"
        ):
            raise GatewayError(502, "oauth_invalid_response", "Invalid authentication response.")
        refresh = tokens.get("refresh_token", old.refresh_token)
        if not isinstance(refresh, str) or not refresh:
            raise GatewayError(502, "oauth_invalid_response", "Missing renewable session.")
        return old.model_copy(
            update={
                "access_token": access,
                "refresh_token": refresh,
                "id_token": tokens.get("id_token", old.id_token),
                "scopes": scopes,
                "expires_at": time.time() + expires,
            }
        )

    async def access_token(self) -> str:
        async with self.store.locked():
            credentials = self.store.read()
            if not credentials or not credentials.access_token or not credentials.client_id:
                raise GatewayError(
                    401, "reauthentication_required", "Run subscription-gateway login."
                )
            if credentials.expires_at <= time.time() + 60:
                if not credentials.refresh_token:
                    raise GatewayError(
                        401, "reauthentication_required", "Run subscription-gateway login."
                    )
                tokens = await self.post_token(
                    {
                        "grant_type": "refresh_token",
                        "client_id": credentials.client_id,
                        "refresh_token": credentials.refresh_token,
                    }
                )
                credentials = self.merge_tokens(credentials, tokens)
                self.store.write(credentials)
            if not REQUIRED_SCOPES.issubset(credentials.scopes):
                raise GatewayError(403, "plan_permission_required", "Authorize ChatGPT plan usage.")
            return credentials.access_token

    async def verify_identity(self, token: str, client_id: str, nonce: str) -> dict:
        try:
            response = await self.client.get(ISSUER + "/.well-known/jwks.json")
            response.raise_for_status()
            keys = jwt.PyJWKSet.from_dict(response.json())
            header = jwt.get_unverified_header(token)
            key = next(key for key in keys.keys if key.key_id == header.get("kid"))
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["RS256"],
                audience=client_id,
                issuer=ISSUER,
                leeway=5,
                options={"require": ["sub", "exp", "iat", "nonce"]},
            )
            if not isinstance(claims["sub"], str) or not claims["sub"]:
                raise ValueError("subject")
            if not hmac.compare_digest(claims["nonce"], nonce):
                raise ValueError("nonce")
            return claims
        except (httpx.HTTPError, jwt.PyJWTError, ValueError, StopIteration, TypeError) as exc:
            raise GatewayError(
                401, "oauth_identity", "Could not validate sign-in identity."
            ) from exc

    async def finish_login(
        self, old: Credentials, attempt: LoginAttempt, params: dict
    ) -> Credentials:
        code, client_id = attempt.callback(params)
        tokens = await self.post_token(
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": attempt.verifier,
                "redirect_uri": attempt.redirect_uri,
            }
        )
        if not isinstance(tokens.get("id_token"), str):
            raise GatewayError(401, "oauth_identity", "Missing sign-in identity.")
        identity = await self.verify_identity(tokens["id_token"], client_id, attempt.nonce)
        if old.subject and identity["sub"] != old.subject:
            raise GatewayError(401, "oauth_identity", "Sign-in returned a different account.")
        new = old.model_copy(update={"client_id": client_id, "subject": identity["sub"]})
        if not isinstance(tokens.get("scope"), str):
            raise GatewayError(
                403, "plan_permission_required", "Sign-in omitted granted plan scopes."
            )
        return self.merge_tokens(new, tokens)

    async def login(self, port: int = 1455) -> None:
        # Holding the session lock keeps login, logout and refresh from overwriting one another.
        async with self.store.locked():
            old = self.store.read() or Credentials(
                ext_agent_host_id="urn:uuid:" + str(uuid.uuid4())
            )
            self.store.write(old)  # Persist host before first authorization.
            loop = asyncio.get_running_loop()
            callback = loop.create_future()
            attempt = LoginAttempt(f"http://127.0.0.1:{port}/auth/callback", old.client_id)

            def deliver(params):
                if not callback.done():
                    callback.set_result(params)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def do_GET(self):
                    parsed = urlparse(self.path)
                    query = parse_qs(parsed.query)
                    valid = (
                        parsed.path == "/auth/callback"
                        and all(len(v) == 1 for v in query.values())
                        and hmac.compare_digest(query.get("state", [""])[0], attempt.state)
                    )
                    self.send_response(200 if valid else 400)
                    self.end_headers()
                    self.wfile.write(b"Return to the terminal." if valid else b"Invalid callback.")
                    if valid:
                        loop.call_soon_threadsafe(deliver, {k: v[0] for k, v in query.items()})

            server = HTTPServer(("127.0.0.1", port), Handler)
            attempt.redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                print("Continue with ChatGPT in your browser.")
                url = attempt.url(old)
                if not webbrowser.open(url):
                    print("Open this private sign-in URL (do not share it):\n" + url)
                params = await asyncio.wait_for(callback, timeout=600)
                new = await self.finish_login(old, attempt, params)
                self.store.write(new)
            except TimeoutError as exc:
                raise GatewayError(400, "oauth_expired", "Sign-in callback timed out.") from exc
            finally:
                await asyncio.to_thread(server.shutdown)
                server.server_close()
                thread.join()

    async def logout(self) -> bool:
        confirmed = True
        async with self.store.locked():
            old = self.store.read()
            if old is None:
                return True
            if old.refresh_token:
                confirmed = False
                try:
                    discovery = await self.client.get(ISSUER + "/.well-known/openid-configuration")
                    discovery.raise_for_status()
                    endpoint = discovery.json()["revocation_endpoint"]
                    if not endpoint.startswith(ISSUER + "/"):
                        raise ValueError("Untrusted revocation endpoint")
                    for delay in (0, 0.5, 1):
                        await asyncio.sleep(delay)
                        response = await self.client.post(
                            endpoint,
                            data={
                                "token": old.refresh_token,
                                "token_type_hint": "refresh_token",
                                "client_id": old.client_id,
                            },
                        )
                        if response.status_code == 200:
                            confirmed = True
                            break
                        if response.status_code < 500:
                            break
                except (httpx.HTTPError, ValueError, KeyError):
                    pass
            self.store.write(
                old.model_copy(
                    update={
                        "access_token": None,
                        "refresh_token": None,
                        "id_token": None,
                        "expires_at": 0,
                    }
                )
            )
        return confirmed
