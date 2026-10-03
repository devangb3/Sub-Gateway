# Subscription Gateway (Python)

A local, single-account gateway that accepts OpenRouter-style Chat Completions
requests and uses an authorized ChatGPT plan through OpenAI’s public Responses API.
It supports text history, function calls/results, JSON responses, and SSE. Clients
supply the whole conversation and execute their own tools.

## Install and run

Python 3.11+ and POSIX file locking (Linux/macOS) are required.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e '.[dev]'
subscription-gateway login
export SUBSCRIPTION_GATEWAY_KEY="$(openssl rand -hex 32)"
subscription-gateway serve
```

The server binds to `127.0.0.1:8787`. Keep the gateway key private and use the same
key in your client. It is separate from your OAuth credentials. No API billing key
is used. `serve` starts the server only when you explicitly run it.

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["SUBSCRIPTION_GATEWAY_KEY"],
    base_url="http://127.0.0.1:8787/v1",
    max_retries=0,
)
models = client.models.list()
model = models.data[0].id  # Select a model from your account’s returned catalog.
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

`GET /health` reports process health without authentication; it does not probe your
account. `GET /v1/models` and `POST /v1/chat/completions` require
`Authorization: Bearer <SUBSCRIPTION_GATEWAY_KEY>`. Model IDs are `openai/<slug>`;
the gateway removes exactly that prefix for inference. Model choices come from the
account catalog, filter `visibility=list`, and cache for 60 seconds. A changed access
token invalidates the cache. Pricing/context limits absent upstream are not invented.

## Authentication route and eligibility

This implements the documented **Sign in with ChatGPT (SIWC), ChatGPT plan usage
for open-source/local apps**, using dynamic client registration. Eligible Plus/Pro
users must grant plan permission; a subscription alone does not prove this
registration/account/model will be accepted. Current access and usage limits are
controlled by OpenAI and the selected account. Review the
[quickstart](https://developers.openai.com/siwc/quickstart) and
[registration contract](https://developers.openai.com/siwc/token-sharing-open-source/sign-in).

First sign-in persists a UUID host ID, registers with `dynamic_agent_client`, and
exchanges the code with the **issued** client ID. Later sign-ins reuse that
registration and verify the same subject. The browser callback is always HTTP on
127.0.0.1 at `/auth/callback`, with state, nonce, S256 PKCE, and a ten-minute expiry.
ID-token signature, issuer, audience, expiry, nonce and subject are validated with
OpenAI’s JWKS. Required inference scopes are `resource.invoke` and
`chatgpt.tokens.use.direct`; identity-only login is rejected.

Inference uses `https://api.openai.com/v1/responses`, always `store:false` and
`stream:true`; even a downstream JSON request aggregates this stream.
[Inference contract](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference).

The referenced Prime files were inspected for authentication, transport and locking.
Prime’s fixed Codex client (`app_EMoamEEZ73f0CkXaXp7hrann`), old OAuth paths and
`chatgpt.com/backend-api` transport are a different integration. This gateway
neither imports nor copies Prime source and never reads its credentials. There is
no legacy adapter or API-key fallback. No copied-code attribution is required.

## Credential ownership and commands

```bash
subscription-gateway status
subscription-gateway logout
subscription-gateway --credentials-dir /your/private/directory login --callback-port 1455
subscription-gateway serve --concurrency 4 --timeout 180 --max-body-bytes 1048576
subscription-gateway serve --cors-origin http://localhost:3000
```

The default store is `~/.subscription-gateway/credentials.json`, distinct from Codex
and Prime stores. Its directory is 0700 and files are 0600. Atomic replacement and
fsync preserve writes; a POSIX flock serializes login/logout/refresh across
processes. Requests reread credentials under the lock and retain rotated refresh
tokens. A lock wait times out after 30 seconds. The CLI must own this directory;
credential and lock symlinks are refused. Do not put the store in a shared folder.

`status` checks local credentials without making network requests. `logout` attempts
session revocation and clears tokens while retaining the host ID and account/client
mapping. If revocation cannot be confirmed, it says so; disconnect the app in
ChatGPT Settings. Requests already in flight may finish; later requests cannot use
the locally cleared tokens. Reauthentication failures require running `login`.
This version has one saved registration; switching to a different account requires
a separate `--credentials-dir` and server restart.

Only loopback bind addresses are accepted. CORS is disabled unless explicit origins
are provided; wildcards are rejected. The default body limit is 1 MiB, body-read
limit 30 seconds, request deadline 180 seconds, concurrency four, event limit 2 MiB,
and aggregate output limit 16 MiB. Full slots produce 429 instead of an unbounded
queue. Uvicorn handles SIGINT/SIGTERM with a ten-second graceful shutdown window.
Access logs are disabled; gateway logs report only fixed error codes/status, not
messages, bodies, tokens, or callback URLs. Client cancellations close upstream
streams. Transport closure is verified offline; remote compute cancellation is not.

## Compatibility matrix

All unknown fields, including unsupported fields with null values, produce HTTP
400. This is a deliberately limited contract, not full OpenRouter parity.

| Field | Behavior |
| --- | --- |
| `model` | Required `openai/<account-visible-slug>`; reversible translation |
| `messages` | Required nonempty ordered history; system/developer/user/assistant/tool |
| `messages[].content` | String or array of `{type:"text",text:...}`; null only for assistant tool calls |
| `messages[].tool_calls` | Assistant function calls; unique IDs, JSON-object argument strings |
| `messages[].tool_call_id` | Tool results must resolve an earlier pending call exactly once |
| `messages[].reasoning_details` | Encrypted `openai-responses-v1` details retained for replay; see below |
| `stream` | Boolean; defaults false; upstream always streams |
| `tools` | Function definitions only: name, description, parameters, strict |
| `tool_choice` | `auto`, `none`, `required`, or `{type:"function",function:{name:...}}` |
| `parallel_tool_calls` | Boolean translated upstream |
| `reasoning.effort` / `reasoning_effort` | One spelling only; none/minimal/low/medium/high/xhigh/max; model support remains upstream-dependent |
| `stream_options` | Only `include_usage:true`; final OpenRouter usage frame is always sent |
| `user`, `metadata` | Harmless client metadata accepted and discarded; not logged or sent upstream |
| Sampling/output controls | temperature, top_p, seed, stop, max_tokens, max_completion_tokens, penalties: rejected |
| Other behavioral fields | response_format, n, logprobs, provider/routing, plugins, models, transforms, service_tier, reasoning summaries/budgets: rejected |
| Other message fields | names, image/audio/file content, annotations, legacy function_call/functions: rejected |

Tool history cannot end with unresolved calls or insert another message before
all pending results arrive. Results may arrive in either order for parallel calls.
The gateway returns function calls; it does not execute them. Callers append the
assistant message and all tool results before requesting the next response.

Provider input/output token totals map directly to prompt/completion tokens.
Cached and reasoning token details are subsets, never added again. Billing costs
are omitted. Unavailable usage is omitted in JSON and null in the SSE accounting
frame. No estimated token counts are fabricated.

Encrypted reasoning is returned under `reasoning_details` with type
`reasoning.encrypted`, format `openai-responses-v1`, item ID, opaque data and output
index. Send these details unchanged with the corresponding assistant message.
The translator reinserts reasoning around text/tool items using those indices.
Reasoning summaries are not exposed; replay supplies an empty summary array.
Chat Completions collapses multiple upstream assistant text items into one string,
so exact replay of unusual interleaved multiple text/reasoning items is not covered.
Opaque reasoning is specific to the upstream model/account and may not be portable.

## Streaming and failures

SSE uses `chat.completion.chunk`, stable gateway ID/created/model, an initial
assistant role, text deltas, and indexed tool identity/argument deltas. A terminal
finish reason is followed by a final usage chunk with **one content-free choice
repeating the finish reason**, then `[DONE]`, as specified by
[OpenRouter’s streaming contract](https://openrouter.ai/docs/api_reference/streaming).
This differs from OpenAI’s empty-choices accounting frame. There is no inference
retry, before or after output. Upstream must emit `response.completed` to succeed;
failed, incomplete and interrupted streams are errors, never successful stops.

Before SSE headers, errors are JSON with the appropriate HTTP status. The first
upstream event is validated before committing those headers. Afterwards errors
include normal chunk metadata, a top-level `error`, and a choice with
`finish_reason:"error"`. Error streams terminate without `[DONE]` or success usage.
Treat such streams as failures even when HTTP status is 200.

| Status / code | Meaning |
| --- | --- |
| 400 `invalid_request` / `invalid_model` / `upstream_rejected` | Invalid fields/history/model or upstream validation failure |
| 401 `invalid_gateway_key` | Wrong local bearer key |
| 401 `reauthentication_required` | Missing/rejected subscription credentials |
| 403 `plan_permission_required` | Plan scopes were not authorized |
| 408 `request_timeout`, 413 `request_too_large` | Body-read limits |
| 429 `gateway_busy` | Local concurrency limit |
| 429 `subscription_limit` / `subscription_sharing_usage_*` | Upstream subscription availability/limits |
| 502 `upstream_unavailable` / `upstream_protocol` / `upstream_interrupted` / `upstream_incomplete` | Provider outage, malformed stream or unsuccessful termination |
| 504 `upstream_timeout` | Provider/request deadline |
| 503 `auth_busy` / `oauth_unavailable` / store errors | Credential contention/auth service/local store failure |

The request/response baseline is the
[OpenRouter Chat Completions reference](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion).
