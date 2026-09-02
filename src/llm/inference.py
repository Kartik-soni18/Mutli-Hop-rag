from cerebras.cloud.sdk import Cerebras
from groq import Groq

from .config import LLMSettings


def generate_groq(prompt: str) -> str:
    settings = LLMSettings()
    client = Groq(api_key=settings.groq_api_key)
    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        reasoning_format="hidden",
        stream=False,
    )
    return completion.choices[0].message.content.strip()


def generate_cerebras(prompt: str) -> str:
    settings = LLMSettings()
    client = Cerebras(api_key=settings.cerebras_cloud_key)
    completion = client.chat.completions.create(
        model=settings.cerebras_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2048,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content.strip()
