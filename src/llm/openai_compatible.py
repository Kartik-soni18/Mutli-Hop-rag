import json
from dataclasses import dataclass
from typing import Sequence

import httpx

from .types import (
    CompletionOptions,
    LLMCapabilities,
    LLMError,
    LLMMessage,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolChoice,
)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    capabilities: LLMCapabilities = LLMCapabilities()
    timeout_seconds: float = 120.0


class OpenAICompatibleLLM:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout_seconds,
        )

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[dict[str, object]] = (),
        tool_choice: ToolChoice | None = None,
        options: CompletionOptions | None = None,
    ) -> LLMResponse:
        options = options or CompletionOptions()
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [message.to_openai_dict() for message in messages],
        }
        if tools:
            if not self.config.capabilities.supports_tools:
                raise LLMError("The configured model does not support tool calling")
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        if options.max_completion_tokens is not None:
            payload["max_completion_tokens"] = options.max_completion_tokens
        if (
            options.reasoning_effort is not None
            and self.config.capabilities.supports_reasoning_effort
        ):
            payload["reasoning_effort"] = options.reasoning_effort
        if (
            options.reasoning_format is not None
            and self.config.capabilities.supports_reasoning_format
        ):
            payload["reasoning_format"] = options.reasoning_format

        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise self._http_error(exc.response) from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"LLM request failed: {exc}",
                retryable=True,
            ) from exc

        return self._normalize_response(response.json())

    @staticmethod
    def _normalize_response(data: dict[str, object]) -> LLMResponse:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("LLM response did not contain any choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMError("LLM response contained an invalid choice")
        raw_message = choice.get("message")
        if not isinstance(raw_message, dict):
            raise LLMError("LLM response did not contain a valid message")

        calls = []
        for raw_call in raw_message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise LLMError("LLM returned invalid tool-call arguments") from exc
            calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                )
            )

        usage_data = data.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(usage_data.get("prompt_tokens") or 0),
            completion_tokens=int(usage_data.get("completion_tokens") or 0),
            total_tokens=int(usage_data.get("total_tokens") or 0),
        )
        return LLMResponse(
            message=LLMMessage(
                role="assistant",
                content=raw_message.get("content"),
                tool_calls=tuple(calls),
            ),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            response_id=data.get("id"),
            model=data.get("model"),
        )

    @staticmethod
    def _http_error(response: httpx.Response) -> LLMError:
        message = f"LLM request failed with status {response.status_code}"
        error_code = None
        try:
            body = response.json()
            error = body.get("error") or {}
            if isinstance(error, dict):
                message = str(
                    error.get("message")
                    or body.get("message")
                    or body.get("detail")
                    or message
                )
                error_code = error.get("code")
            else:
                message = str(error or body.get("message") or message)
        except (ValueError, AttributeError):
            pass
        return LLMError(
            message,
            status_code=response.status_code,
            error_code=error_code,
            retryable=response.status_code == 429 or response.status_code >= 500,
        )

    def close(self) -> None:
        self._client.close()
