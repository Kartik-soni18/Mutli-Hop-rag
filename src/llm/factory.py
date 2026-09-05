import os
from dataclasses import dataclass
from functools import cache

from dotenv import load_dotenv

from .interface import LLM
from .openai_compatible import OpenAICompatibleConfig, OpenAICompatibleLLM
from .types import LLMCapabilities, LLMError


load_dotenv()


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    base_url_environment_variable: str
    default_base_url: str
    api_key_environment_variable: str
    model_environment_variable: str
    default_model: str
    capabilities: LLMCapabilities


OPENAI_COMPATIBLE_PROVIDERS = {
    "groq": ProviderDefinition(
        base_url_environment_variable="GROQ_BASE_URL",
        default_base_url="https://api.groq.com/openai/v1",
        api_key_environment_variable="GROQ_API_KEY",
        model_environment_variable="GROQ_MODEL",
        default_model="openai/gpt-oss-120b",
        capabilities=LLMCapabilities(
            supports_tools=True,
            supports_reasoning_effort=True,
            supports_reasoning_format=True,
        ),
    ),
    "cerebras": ProviderDefinition(
        base_url_environment_variable="CEREBRAS_BASE_URL",
        default_base_url="https://api.cerebras.ai/v1",
        api_key_environment_variable="CEREBRAS_CLOUD_KEY",
        model_environment_variable="CEREBRAS_MODEL",
        default_model="gpt-oss-120b",
        capabilities=LLMCapabilities(
            supports_tools=True,
            supports_reasoning_effort=True,
            supports_reasoning_format=True,
        ),
    ),
}


@cache
def create_llm() -> LLM:
    provider_name = os.getenv("LLM_PROVIDER", "groq").strip().casefold()
    try:
        provider = OPENAI_COMPATIBLE_PROVIDERS[provider_name]
    except KeyError as exc:
        supported = ", ".join(sorted(OPENAI_COMPATIBLE_PROVIDERS))
        raise LLMError(
            f"Unknown LLM_PROVIDER {provider_name!r}; expected one of: {supported}"
        ) from exc

    api_key = os.getenv(provider.api_key_environment_variable)
    if not api_key:
        raise LLMError(
            f"Missing environment variable {provider.api_key_environment_variable}"
        )

    return OpenAICompatibleLLM(
        OpenAICompatibleConfig(
            base_url=os.getenv(
                provider.base_url_environment_variable,
                provider.default_base_url,
            ),
            api_key=api_key,
            model=os.getenv(
                provider.model_environment_variable,
                provider.default_model,
            ),
            capabilities=provider.capabilities,
        )
    )
