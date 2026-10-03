import asyncio
import base64
import hashlib
import json
import time
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from subscription_gateway.auth.oauth import ISSUER, LoginAttempt, OAuthManager
from subscription_gateway.auth.store import Credentials, FileCredentialStore
from subscription_gateway.errors import GatewayError


def saved():
    return Credentials(
        ext_agent_host_id="urn:uuid:host",
        client_id="issued",
        subject="subject",
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at=1,
        scopes=["resource.invoke", "chatgpt.tokens.use.direct"],
    )


async def test_concurrent_refresh_and_rotation(tmp_path):
    directory = tmp_path / "credentials"
    store = FileCredentialStore(directory)
    async with store.locked():
        store.write(saved())
    calls = []

    async def handler(request):
        calls.append(parse_qs(request.content.decode()))
        await asyncio.sleep(0.02)
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "rotated",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        managers = [OAuthManager(FileCredentialStore(directory), client) for _ in range(12)]
        tokens = await asyncio.gather(*(manager.access_token() for manager in managers))
    assert tokens == ["new-access"] * 12
    assert len(calls) == 1
    assert calls[0]["client_id"] == ["issued"]
    assert calls[0]["resource"] == ["https://api.openai.com/v1"]
    assert "scope" not in calls[0]
    assert store.read().refresh_token == "rotated"
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    assert not list(directory.glob(".credentials-*"))


async def test_lock_between_processes(tmp_path):
    import sys

    store = FileCredentialStore(tmp_path / "credentials")
    script = """
import asyncio, sys
from pathlib import Path
from subscription_gateway.auth.store import FileCredentialStore
async def run():
    async with FileCredentialStore(Path(sys.argv[1])).locked():
        print('acquired', flush=True)
asyncio.run(run())
"""
    async with store.locked():
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, str(store.directory), stdout=asyncio.subprocess.PIPE
        )
        reader = asyncio.create_task(child.stdout.readline())
        await asyncio.sleep(0.15)
        assert not reader.done()
    assert await asyncio.wait_for(reader, timeout=3) == b"acquired\n"
    assert await child.wait() == 0


def test_pkce_state_expiry_and_registration():
    attempt = LoginAttempt("http://127.0.0.1:1455/auth/callback")
    params = parse_qs(urlparse(attempt.url(saved())).query)
    assert params["client_id"] == ["dynamic_agent_client"]
    expected = base64.urlsafe_b64encode(hashlib.sha256(attempt.verifier.encode()).digest())
    assert params["code_challenge"] == [expected.decode().rstrip("=")]
    with pytest.raises(GatewayError, match="Invalid OAuth state"):
        attempt.callback({"state": "bad", "code": "code", "client_id": "issued"})
    assert not attempt.consumed
    assert attempt.callback({"state": attempt.state, "code": "code", "client_id": "issued"}) == (
        "code",
        "issued",
    )
    with pytest.raises(GatewayError):
        attempt.callback({"state": attempt.state, "code": "code", "client_id": "issued"})
    expired = LoginAttempt("http://127.0.0.1:1455/auth/callback", expires_at=0)
    with pytest.raises(GatewayError):
        expired.callback({"state": expired.state})
    returning = LoginAttempt("http://127.0.0.1:1455/auth/callback", "issued")
    assert "agent_name_hint" not in parse_qs(urlparse(returning.url(saved())).query)
    with pytest.raises(GatewayError):
        returning.callback({"state": returning.state, "code": "code", "client_id": "other"})


@pytest.mark.parametrize(
    "invalid", ["nonce", "audience", "issuer", "expired", "subject", "signature", "scope"]
)
async def test_signed_identity_validation(tmp_path, invalid):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    public["kid"] = "test-key"
    attempt = LoginAttempt("http://127.0.0.1:1455/auth/callback", "issued")
    claims = {
        "sub": "subject",
        "iss": ISSUER,
        "aud": "issued",
        "nonce": attempt.nonce,
        "exp": int(time.time()) + 600,
        "iat": int(time.time()),
    }
    if invalid == "nonce":
        claims["nonce"] = "other"
    elif invalid == "audience":
        claims["aud"] = "other"
    elif invalid == "issuer":
        claims["iss"] = "https://evil.example"
    elif invalid == "expired":
        claims["exp"] = 1
    elif invalid == "subject":
        claims["sub"] = "another-account"
    signing_key = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        if invalid == "signature"
        else private_key
    )
    encoded = jwt.encode(claims, signing_key, algorithm="RS256", headers={"kid": "test-key"})

    async def handler(request):
        if request.url.path.endswith("jwks.json"):
            return httpx.Response(200, json={"keys": [public]})
        return httpx.Response(
            200,
            json={
                "id_token": encoded,
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": 3600,
                "scope": "openid"
                if invalid == "scope"
                else "openid resource.invoke chatgpt.tokens.use.direct",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        oauth = OAuthManager(FileCredentialStore(tmp_path / "auth"), client)
        with pytest.raises(GatewayError):
            await oauth.finish_login(saved(), attempt, {"code": "code", "state": attempt.state})


async def test_successful_registration_and_logout(tmp_path):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    public["kid"] = "key"
    attempt = LoginAttempt("http://127.0.0.1:1455/auth/callback")
    encoded = jwt.encode(
        {
            "iss": ISSUER,
            "aud": "new-issued",
            "sub": "subject",
            "nonce": attempt.nonce,
            "iat": int(time.time()),
            "exp": int(time.time()) + 600,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "key"},
    )
    requests = []

    async def handler(request):
        requests.append(request)
        if request.url.path.endswith("jwks.json"):
            return httpx.Response(200, json={"keys": [public]})
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(200, json={"revocation_endpoint": ISSUER + "/revoke"})
        if request.url.path == "/revoke":
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={
                "id_token": encoded,
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": 3600,
                "scope": "openid resource.invoke chatgpt.tokens.use.direct",
            },
        )

    store = FileCredentialStore(tmp_path / "auth")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        oauth = OAuthManager(store, client)
        old = Credentials(ext_agent_host_id="urn:uuid:host")
        new = await oauth.finish_login(
            old, attempt, {"code": "code", "state": attempt.state, "client_id": "new-issued"}
        )
        async with store.locked():
            store.write(new)
        assert new.subject == "subject"
        fields = parse_qs(requests[0].content.decode())
        assert fields["client_id"] == ["new-issued"]
        assert fields["redirect_uri"] == [attempt.redirect_uri]
        assert fields["code_verifier"] == [attempt.verifier]
        assert await oauth.logout()
        remaining = store.read()
        assert remaining.access_token is remaining.refresh_token is remaining.id_token is None
        assert remaining.client_id == "new-issued"
        assert remaining.ext_agent_host_id == old.ext_agent_host_id


async def test_refresh_failure_and_unconfirmed_revocation(tmp_path):
    store = FileCredentialStore(tmp_path / "auth")
    async with store.locked():
        store.write(saved())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(400, json={"error": "invalid_grant", "private": "SECRET"})
        )
    ) as client:
        oauth = OAuthManager(store, client)
        with pytest.raises(GatewayError) as error:
            await oauth.access_token()
        assert error.value.code == "reauthentication_required"
        assert "SECRET" not in str(error.value)
        assert store.read().refresh_token == "old-refresh"
        assert not await oauth.logout()
        assert store.read().access_token is None


async def test_refresh_serialized_across_processes(tmp_path):
    import sys

    store = FileCredentialStore(tmp_path / "auth")
    async with store.locked():
        store.write(saved())
    count_file = tmp_path / "refresh-count"
    script = """
import asyncio, sys
from pathlib import Path
import httpx
from subscription_gateway.auth.oauth import OAuthManager
from subscription_gateway.auth.store import FileCredentialStore
async def handler(request):
    with Path(sys.argv[2]).open('a') as handle:
        handle.write('refreshed\\n')
    await asyncio.sleep(0.1)
    return httpx.Response(200, json={'access_token':'new',
                                    'refresh_token':'rotated', 'expires_in':3600})
async def run():
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        token = await OAuthManager(FileCredentialStore(Path(sys.argv[1])), client).access_token()
        assert token == 'new'
asyncio.run(run())
"""
    children = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            str(store.directory),
            str(count_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for _ in range(3)
    ]
    for child in children:
        _, stderr = await asyncio.wait_for(child.communicate(), 5)
        assert child.returncode == 0, stderr.decode()
    assert count_file.read_text() == "refreshed\n"
    assert store.read().refresh_token == "rotated"
