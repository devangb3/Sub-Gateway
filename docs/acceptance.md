# Acceptance evidence

Verified on 2026-10-02 in `/home/devang/Projects/codex-server`, Python 3.13.9.
The workspace began empty; the implementation is a standalone Python package.

`./scripts/check.sh` completed successfully. It activated `.venv` and ran:

```text
ruff check src tests scripts/live_smoke.py
All checks passed!
ruff format --check src tests scripts/live_smoke.py
21 files already formatted
pytest -q tests/test_auth.py
13 passed in 1.01s
pytest -q tests/test_contract.py
22 passed in 0.96s
pytest -q tests/test_transport.py
12 passed in 0.03s
pytest -q tests/test_operations.py
8 passed in 0.34s
python -m subscription_gateway.cli --help
[CLI help printed successfully]
```

Total: **55 passing offline tests**. No TypeScript workspace or repository-root
npm check exists here; the Python check script replaces the plan’s Node commands.
Dependencies were installed from the local uv cache because network access to PyPI
was unavailable. A dependency lock was also resolved from cached metadata. Tests
ran in the installed local environment (FastAPI 0.141.1, httpx 0.28.1, Pydantic
2.13.4, PyJWT 2.13.0, OpenAI SDK 2.50.0, Uvicorn 0.52.1).

| Area | Evidence boundary |
| --- | --- |
| OAuth registration/code exchange | Scripted token endpoint and real RSA-signed ID tokens/JWKS; validates PKCE/state/expiry, issuer/audience/nonce/subject and grant scope |
| Refresh and file safety | Atomic private files, 12 concurrent manager instances, independent-process locking, and three separate processes sharing one rotating refresh |
| Logout | Scripted discovery/revocation, locally cleared tokens and retained registration/host ID; failure is reported as unconfirmed revocation |
| Catalog | Mock HTTP account catalog, visibility filtering, stable IDs and cache reuse |
| JSON/SSE | Same scripted upstream events through the actual ASGI app; stable chunk metadata, text/tool reconstruction, terminal usage and failure envelopes |
| Tool history | Parallel calls and results in reverse order, preserved call IDs, named selection, malformed-history rejection and encrypted reasoning replay |
| UTF-8/SSE | Byte-fragmented text including emoji, multiline data, comments, CR/LF/CRLF, malformed JSON/UTF-8, partial events and bounded records |
| Cancellation/backpressure | Mock transport closure, direct ASGI disconnect, blocked downstream writes, slot release and before/after-header timeouts |
| Operational controls | Bearer protection, unknown parameter rejection, body limits, bounded concurrency and explicit CORS |
| OpenAI SDK | `AsyncOpenAI` JSON and streaming against in-process ASGI transport; SDK host platform probe fixed to Linux for the restricted test environment |

Live verification: **not performed**. No account credentials were supplied or read
from Codex/Prime, no browser login was launched, and no inference request consumed
subscription usage. The gateway server was not started. Real socket behavior,
browser OAuth callback delivery, current account eligibility, model capabilities,
remote cancellation and subscription limits remain outside the offline evidence.
The CLI contains Uvicorn graceful-shutdown configuration; signal behavior has not
been measured with a running server.

The opt-in live procedure in README and `scripts/live_smoke.py --live --model ...`
checks health, models, JSON, SSE/provider usage, and a function-call/result round
trip after explicit user login/startup. Failure of that procedure must be reported
as a live failure, without paid API-key fallback.
