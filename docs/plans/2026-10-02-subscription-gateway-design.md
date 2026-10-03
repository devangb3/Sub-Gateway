# Python subscription gateway

Implement the supplied plan using Python 3.11+, FastAPI/Pydantic external schemas,
httpx streaming transport, and a separate POSIX credential store. Use the documented
SIWC dynamic registration flow and public Responses API only. No API-key fallback
or Prime adapter is needed. One process may serve bounded concurrent requests;
flock serializes credential changes and refreshes across processes.

The gateway consumes Responses SSE in both JSON and streaming modes. A shared
normalizer translates ordered text/tool events, final usage and encrypted reasoning
items. HTTP streaming pulls directly from upstream for backpressure and closes the
upstream iterator when cancelled. No inference retries. Validate the first event
before sending SSE headers; after headers, errors become an SSE error envelope.

## Deliverables

1. [x] Verify official upstream route and inspect Prime reference integration.
2. [x] Python package, typed boundaries and validated schemas.
3. [x] Health, models, completions and compatibility matrix.
4. [x] OAuth CLI and atomic, separately owned credentials.
5. [x] Account-specific models and ordered request translation.
6. [x] JSON aggregation, SSE normalization, usage and reasoning replay.
7. [x] Access controls, limits, cancellation and failure handling.
8. [x] Offline contract tests and OpenAI SDK checks.
9. [x] Setup, changelog and opt-in live acceptance procedure.

The TypeScript vitest/npm commands in the original plan do not apply to this empty
Python workspace. Run each new Python test file separately and the Python check
script; do not start a development server or make live account requests.
