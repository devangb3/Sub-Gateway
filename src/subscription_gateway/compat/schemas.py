import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TextPart(StrictModel):
    type: Literal["text"]
    text: str


class CallFunction(StrictModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    arguments: str

    @model_validator(mode="after")
    def valid_arguments(self):
        try:
            if not isinstance(json.loads(self.arguments), dict):
                raise ValueError()
        except ValueError as exc:
            raise ValueError("Function arguments must encode a JSON object") from exc
        return self


class ToolCall(StrictModel):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: CallFunction


class ReasoningDetail(StrictModel):
    type: Literal["reasoning.encrypted"]
    format: Literal["openai-responses-v1"]
    id: str = Field(min_length=1)
    data: str = Field(min_length=1)
    index: int = Field(default=0, ge=0)


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    reasoning_details: list[ReasoningDetail] | None = None

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content or [])

    @model_validator(mode="after")
    def role_fields(self):
        if self.role != "assistant" and (
            self.tool_calls is not None or self.reasoning_details is not None
        ):
            raise ValueError("Only assistant messages may contain calls/reasoning")
        if self.role == "tool":
            if not self.tool_call_id or self.content is None:
                raise ValueError("Tool results require tool_call_id and text content")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid on tool results")
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError("Message requires text content or assistant tool calls")
        return self


class FunctionDefinition(StrictModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool | None = None


class ToolDefinition(StrictModel):
    type: Literal["function"]
    function: FunctionDefinition


class NamedFunction(StrictModel):
    name: str


class NamedToolChoice(StrictModel):
    type: Literal["function"]
    function: NamedFunction


Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class Reasoning(StrictModel):
    effort: Effort


class StreamOptions(StrictModel):
    include_usage: bool = True


class ChatRequest(StrictModel):
    model: str = Field(pattern=r"^openai/[^/\s]+$")
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    reasoning: Reasoning | None = None
    reasoning_effort: Effort | None = None
    stream_options: StreamOptions | None = None
    user: str | None = None
    metadata: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_conversation(self):
        if self.reasoning and self.reasoning_effort:
            raise ValueError("Supply reasoning.effort or reasoning_effort, not both")
        if self.stream_options and not self.stream:
            raise ValueError("stream_options requires stream=true")
        if self.stream_options and not self.stream_options.include_usage:
            raise ValueError("OpenRouter compatibility always includes final usage")
        names = [tool.function.name for tool in self.tools or []]
        if len(names) != len(set(names)):
            raise ValueError("Tool names must be unique")
        if self.tool_choice not in (None, "auto", "none") and not names:
            raise ValueError("tool_choice requires tools")
        if (
            isinstance(self.tool_choice, NamedToolChoice)
            and self.tool_choice.function.name not in names
        ):
            raise ValueError("Selected function is not in tools")
        pending: set[str] = set()
        seen: set[str] = set()
        for message in self.messages:
            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("Tool result has no matching pending function call")
                pending.remove(message.tool_call_id)
            else:
                if pending:
                    raise ValueError("All function calls need tool results before the next message")
                for call in message.tool_calls or []:
                    if call.id in seen:
                        raise ValueError("Function call IDs must be unique")
                    seen.add(call.id)
                    pending.add(call.id)
        if pending:
            raise ValueError("Conversation ends with unresolved function calls")
        return self
