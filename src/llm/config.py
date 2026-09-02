import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True, slots=True)
class LLMSettings:
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    cerebras_cloud_key: str | None = os.getenv("CEREBRAS_CLOUD_KEY")
    cerebras_model: str = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
