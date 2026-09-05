from typing import Protocol, Sequence

from .types import CompletionOptions, LLMMessage, LLMResponse, ToolChoice


class LLM(Protocol):
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[dict[str, object]] = (),
        tool_choice: ToolChoice | None = None,
        options: CompletionOptions | None = None,
    ) -> LLMResponse: ...

