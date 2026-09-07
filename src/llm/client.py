"""AI Credits Chat Completions through LangChain and the OpenAI SDK."""

import os
from dataclasses import dataclass
from functools import cache
from time import perf_counter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.observability.pricing import price
from src.observability.tracing import enabled, observation, record_llm, safe


class AICreditsChat(ChatOpenAI):
    """Keep only billing extensions that standard ChatOpenAI would discard."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump()
        usage = data.get("usage") or {}
        billing = {}
        for source in (usage, data):
            if source.get("cost_usd") is not None:
                billing = {"amount": source["cost_usd"], "currency": "USD"}
            elif source.get("cost") is not None and source.get("currency"):
                billing = {"amount": source["cost"], "currency": source["currency"]}
        result.llm_output["provider_billing"] = billing
        # Do not substitute the requested model when the response omits it.
        result.llm_output["model_name"] = data.get("model")
        return result


@dataclass(frozen=True)
class Config:
    model: str
    base_url: str


class AICreditsLLM:
    def __init__(self):
        load_dotenv()
        key = os.getenv("AI_CREDITS")
        if not key:
            raise ValueError("Missing AI_CREDITS")
        self.config = Config(
            os.getenv("AICREDITS_MODEL", "inclusionai/ling-3.0-flash"),
            os.getenv("AICREDITS_BASE_URL", "https://api.aicredits.in/v1"),
        )
        self.model = AICreditsChat(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=key,
            max_retries=0,
            timeout=120,
            streaming=False,
            use_responses_api=False,
        )

    def complete(
        self, messages, *, purpose, tools=None, max_tokens=512, context_ids=None
    ):
        with observation(
            f"llm_{purpose}",
            "generation",
            provider="aicredits",
            requested_model=self.config.model,
            purpose=purpose,
            retry_count=0,
            time_to_first_token=None,
            context_ids=context_ids or [],
        ) as obs:
            prompt_chars = sum(len(m["content"]) for m in messages)
            obs.update(prompt_size_chars=prompt_chars)
            if enabled("prompts") and (
                purpose != "final_answer" or enabled("chunk_text")
            ):
                obs.payload(input=messages)
            runnable = self.model
            if tools:
                runnable = runnable.bind_tools(tools, tool_choice="required")
            started = perf_counter()
            response = runnable.invoke(messages, temperature=0, max_tokens=max_tokens)
            raw = response.response_metadata.get("token_usage") or {}
            usage = {
                "input_tokens": raw.get("prompt_tokens"),
                "output_tokens": raw.get("completion_tokens"),
                "total_tokens": raw.get("total_tokens"),
                "cached_tokens": (raw.get("prompt_tokens_details") or {}).get(
                    "cached_tokens"
                ),
            }
            metadata = dict(
                usage,
                actual_response_model=response.response_metadata.get("model_name"),
                finish_reason=response.response_metadata.get("finish_reason"),
                latency_ms=(perf_counter() - started) * 1000,
                response_length=len(response.content),
                prompt_tokens=usage["input_tokens"],
            )
            pricing = safe(
                price,
                usage,
                self.config.model,
                metadata["actual_response_model"],
                response.response_metadata.get("provider_billing"),
            )
            metadata.update(
                pricing
                or {"cost_usd": None, "cost_source": "unavailable", "currency": "USD"}
            )
            obs.update(**metadata)
            safe(record_llm, dict(obs.metadata))
            native_usage = {
                k: v
                for k, v in {
                    "input": usage["input_tokens"],
                    "output": usage["output_tokens"],
                    "total": usage["total_tokens"],
                }.items()
                if v is not None
            }
            obs.payload(
                model=metadata["actual_response_model"], usage_details=native_usage
            )
            if metadata["cost_usd"] is not None:
                obs.payload(cost_details={"total": float(metadata["cost_usd"])})
            if enabled("responses"):
                obs.payload(
                    output={
                        "content": response.content,
                        "tool_calls": response.tool_calls,
                    }
                )
            return response

    def close(self):
        self.model.root_client.close()


@cache
def create_llm():
    return AICreditsLLM()
