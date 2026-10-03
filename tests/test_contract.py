import asyncio
import json

import pytest
from conftest import KEY, MODEL, parse_chunks
from openai import AsyncOpenAI

from subscription_gateway.compat.requests import translate
from subscription_gateway.compat.schemas import ChatRequest


async def test_json_sse_equivalence_and_usage(fake_gateway):
    client, state, _ = fake_gateway
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hello"}]}
    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    completion = response.json()
    response = await client.post("/v1/chat/completions", json={**payload, "stream": True})
    assert response.status_code == 200
    chunks = parse_chunks(response.text)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert len({chunk["created"] for chunk in chunks}) == 1
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == completion["choices"][0]["message"]["content"] == "Hello 🌍"
    assert chunks[-1]["usage"] == completion["usage"]
    assert completion["usage"]["total_tokens"] == 30
    assert completion["usage"]["completion_tokens_details"]["reasoning_tokens"] == 6
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert response.text.endswith("data: [DONE]\n\n")
    assert state["requests"][0]["stream"] is True
    assert state["requests"][0]["store"] is False
    assert all(stream.closed for stream in state["streams"])


async def test_models_cache_auth_and_health(fake_gateway):
    client, state, _ = fake_gateway
    first = await client.get("/v1/models")
    assert [model["id"] for model in first.json()["data"]] == [MODEL]
    await client.get("/v1/models")
    assert state["catalog_calls"] == 1
    assert (await client.get("/health", headers={"Authorization": "bad"})).status_code == 200
    assert (await client.get("/v1/models", headers={"Authorization": "bad"})).status_code == 401
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "openai/bad", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_model"


@pytest.mark.parametrize(
    "extra",
    [
        {"temperature": 0.1},
        {"max_tokens": 20},
        {"stop": ["x"]},
        {"provider": {}},
        {"n": 2},
        {"response_format": {}},
        {"stream": "false"},
        {"reasoning": {"effort": "invalid"}},
        {"stream_options": {"include_usage": False}, "stream": True},
    ],
)
async def test_unsupported_parameters(fake_gateway, extra):
    client, state, _ = fake_gateway
    response = await client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hello"}], **extra},
    )
    assert response.status_code == 400
    assert not state["requests"]


async def test_openai_sdk_json_and_stream(fake_gateway):
    client, _, _ = fake_gateway
    sdk = AsyncOpenAI(api_key=KEY, base_url="http://gateway/v1", http_client=client, max_retries=0)
    # Avoid SDK host probing through a thread pool in this restricted test environment.
    sdk._platform = "Linux"
    result = await asyncio.wait_for(
        sdk.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}]), 5
    )
    assert result.choices[0].message.content == "Hello 🌍"
    stream = await asyncio.wait_for(
        sdk.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hi"}], stream=True
        ),
        5,
    )
    chunks = [chunk async for chunk in stream]
    assert chunks[-1].usage.total_tokens == 30
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_ordered_translation_and_reasoning():
    request = ChatRequest.model_validate(
        {
            "model": MODEL,
            "reasoning": {"effort": "high"},
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "developer", "content": "dev"},
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
                {
                    "role": "assistant",
                    "content": "checking",
                    "reasoning_details": [
                        {
                            "type": "reasoning.encrypted",
                            "format": "openai-responses-v1",
                            "id": "rs1",
                            "data": "opaque",
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "call1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call1", "content": "result"},
            ],
        }
    )
    upstream = translate(request)
    assert [item.get("type", item.get("role")) for item in upstream["input"]] == [
        "system",
        "developer",
        "user",
        "reasoning",
        "assistant",
        "function_call",
        "function_call_output",
    ]
    assert upstream["input"][-2]["call_id"] == upstream["input"][-1]["call_id"]
    assert upstream["tool_choice"] == {"type": "function", "name": "lookup"}
    assert upstream["reasoning"] == {"effort": "high"}


async def test_multi_tool_round_trip(fake_gateway):
    client, state, _ = fake_gateway
    calls = [
        {
            "type": "function_call",
            "id": "fc" + str(index),
            "call_id": "call" + str(index),
            "name": "lookup",
            "arguments": json.dumps({"q": index}),
        }
        for index in range(2)
    ]
    state["events"] = [{"type": "response.created"}]
    for index, call in enumerate(calls):
        state["events"] += [
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {**call, "arguments": ""},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": index,
                "delta": call["arguments"][:3],
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": index,
                "delta": call["arguments"][3:],
            },
        ]
    state["events"] += [
        {
            "type": "response.completed",
            "response": {"output": calls, "usage": {"input_tokens": 5, "output_tokens": 9}},
        }
    ]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "lookup"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    json_response = (await client.post("/v1/chat/completions", json=payload)).json()
    assistant = json_response["choices"][0]["message"]
    assert len(assistant["tool_calls"]) == 2
    assert json_response["choices"][0]["finish_reason"] == "tool_calls"
    streamed = await client.post("/v1/chat/completions", json={**payload, "stream": True})
    reconstructed = {}
    for chunk in parse_chunks(streamed.text):
        for call in chunk["choices"][0]["delta"].get("tool_calls", []):
            entry = reconstructed.setdefault(call["index"], {"arguments": ""})
            entry.update({key: value for key, value in call.items() if key != "function"})
            entry["arguments"] += call.get("function", {}).get("arguments", "")
    assert [entry["arguments"] for entry in reconstructed.values()] == [
        c["arguments"] for c in calls
    ]
    payload["messages"] += [
        assistant,
        {"role": "tool", "tool_call_id": "call1", "content": "one"},
        {"role": "tool", "tool_call_id": "call0", "content": "zero"},
    ]
    assert (await client.post("/v1/chat/completions", json=payload)).status_code == 200
    assert [item["call_id"] for item in state["requests"][-1]["input"][-2:]] == ["call1", "call0"]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "tool_call_id": "missing", "content": "x"}],
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": "{}"}}],
            }
        ],
    ],
)
async def test_malformed_history(fake_gateway, messages):
    client, _, _ = fake_gateway
    assert (
        await client.post("/v1/chat/completions", json={"model": MODEL, "messages": messages})
    ).status_code == 400


async def test_before_and_after_headers(fake_gateway):
    client, state, _ = fake_gateway
    payload = {"model": MODEL, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    state["status"] = 429
    before = await client.post("/v1/chat/completions", json=payload)
    assert before.status_code == 429
    assert "SECRET" not in before.text
    state["status"] = 200
    state["events"] = [
        {
            "type": "response.failed",
            "response": {
                "error": {"code": "subscription_sharing_usage_limit_exceeded", "message": "SECRET"}
            },
        }
    ]
    assert (await client.post("/v1/chat/completions", json=payload)).status_code == 429
    state["events"].insert(0, {"type": "response.created"})
    after = await client.post("/v1/chat/completions", json=payload)
    assert after.status_code == 200
    chunk = parse_chunks(after.text)[-1]
    assert chunk["error"]["code"] == "subscription_sharing_usage_limit_exceeded"
    assert chunk["choices"][0]["finish_reason"] == "error"
    assert "[DONE]" not in after.text
    assert "SECRET" not in after.text


async def test_interrupted_stream_is_failure(fake_gateway):
    client, state, _ = fake_gateway
    state["events"] = state["events"][:-1]
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
    assert (await client.post("/v1/chat/completions", json=payload)).status_code == 502
    response = await client.post("/v1/chat/completions", json={**payload, "stream": True})
    assert parse_chunks(response.text)[-1]["error"]["code"] == "upstream_interrupted"
    assert "[DONE]" not in response.text


async def test_reasoning_replay_and_unavailable_cost(fake_gateway):
    client, state, _ = fake_gateway
    state["events"][-1]["response"]["output"].insert(
        0,
        {"type": "reasoning", "id": "rs1", "encrypted_content": "encrypted-opaque", "summary": []},
    )
    # Move text index with the inserted reasoning item.
    for event in state["events"]:
        if "output_index" in event:
            event["output_index"] = 1
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
    response = (await client.post("/v1/chat/completions", json=payload)).json()
    assert "cost" not in response["usage"]
    assistant = response["choices"][0]["message"]
    payload["messages"] += [assistant, {"role": "user", "content": "again"}]
    assert (await client.post("/v1/chat/completions", json=payload)).status_code == 200
    assert state["requests"][-1]["input"][1]["encrypted_content"] == "encrypted-opaque"


async def test_completion_only_upstream_and_incomplete(fake_gateway):
    client, state, _ = fake_gateway
    state["events"] = [state["events"][-1]]
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True}
    response = await client.post("/v1/chat/completions", json=payload)
    chunks = parse_chunks(response.text)
    assert (
        "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == "Hello 🌍"
    )
    state["events"] = [{"type": "response.created"}, {"type": "response.incomplete"}]
    response = await client.post("/v1/chat/completions", json=payload)
    assert parse_chunks(response.text)[-1]["error"]["code"] == "upstream_incomplete"
    assert "[DONE]" not in response.text


def test_reasoning_between_function_calls_preserves_order():
    request = ChatRequest.model_validate(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "a", "function": {"name": "f", "arguments": "{}"}},
                        {"id": "b", "function": {"name": "f", "arguments": "{}"}},
                    ],
                    "reasoning_details": [
                        {
                            "type": "reasoning.encrypted",
                            "format": "openai-responses-v1",
                            "id": "rs1",
                            "data": "opaque1",
                            "index": 0,
                        },
                        {
                            "type": "reasoning.encrypted",
                            "format": "openai-responses-v1",
                            "id": "rs2",
                            "data": "opaque2",
                            "index": 2,
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "one"},
                {"role": "tool", "tool_call_id": "b", "content": "two"},
            ],
        }
    )
    assert [item["type"] for item in translate(request)["input"]] == [
        "reasoning",
        "function_call",
        "reasoning",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
