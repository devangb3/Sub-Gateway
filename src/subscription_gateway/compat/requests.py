from subscription_gateway.compat.schemas import ChatRequest, NamedToolChoice
from subscription_gateway.providers.base import ProviderRequest


def translate(request: ChatRequest) -> ProviderRequest:
    items: list[dict] = []
    for message in request.messages:
        if message.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.text(),
                }
            )
            continue
        turn: list[dict] = []
        if message.content is not None:
            kind = "output_text" if message.role == "assistant" else "input_text"
            turn.append({"role": message.role, "content": [{"type": kind, "text": message.text()}]})
        for call in message.tool_calls or []:
            turn.append(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
            )
        # Encrypted details retain upstream output indices. Reinsert around
        # assistant text and function calls, including reasoning between calls.
        for detail in sorted(message.reasoning_details or [], key=lambda item: item.index):
            turn.insert(
                detail.index,
                {
                    "type": "reasoning",
                    "id": detail.id,
                    "encrypted_content": detail.data,
                    "summary": [],
                },
            )
        items.extend(turn)
    result: ProviderRequest = {
        "model": request.model.removeprefix("openai/"),
        "input": items,
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }
    if request.tools is not None:
        result["tools"] = [
            {"type": "function", **tool.function.model_dump(exclude_none=True)}
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        choice = request.tool_choice
        result["tool_choice"] = (
            {"type": "function", "name": choice.function.name}
            if isinstance(choice, NamedToolChoice)
            else choice
        )
    if request.parallel_tool_calls is not None:
        result["parallel_tool_calls"] = request.parallel_tool_calls
    effort = request.reasoning.effort if request.reasoning else request.reasoning_effort
    if effort:
        result["reasoning"] = {"effort": effort}
    return result
