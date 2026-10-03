from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, Required, TypedDict


class ProviderRequest(TypedDict, total=False):
    model: Required[str]
    input: Required[list[dict[str, Any]]]
    store: Required[Literal[False]]
    stream: Required[Literal[True]]
    include: list[str]
    tools: list[dict[str, Any]]
    tool_choice: str | dict[str, str]
    parallel_tool_calls: bool
    reasoning: dict[str, str]


class ProviderEvent(TypedDict, total=False):
    type: str
    delta: str
    output_index: int
    item: dict[str, Any]
    response: dict[str, Any]


class Provider(Protocol):
    async def models(self) -> list[dict[str, Any]]: ...
    def events(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]: ...
    async def aclose(self) -> None: ...
