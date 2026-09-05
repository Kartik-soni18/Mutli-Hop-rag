import json
from dataclasses import dataclass, field
from typing import Literal


MessageRole = Literal["system", "user", "assistant", "tool"]
ToolChoice = Literal["auto", "none", "required"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai_dict(self) -> dict[str, object]:
        message: dict[str, object] = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            message["name"] = self.name
        return message


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    message: LLMMessage
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    response_id: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionOptions:
    temperature: float | None = None
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None
    reasoning_format: str | None = None


@dataclass(frozen=True, slots=True)
class LLMCapabilities:
    supports_tools: bool = True
    supports_reasoning_effort: bool = False
    supports_reasoning_format: bool = False


class LLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable

