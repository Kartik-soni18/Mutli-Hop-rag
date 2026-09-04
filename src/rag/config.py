from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True, slots=True)
class Settings:

    corpus_path: Path = BASE_DIR / "data" / "corpus.json"
    qdrant_path: Path = BASE_DIR / "qdrant_db"
    collection_name: str = "multihop_rag"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model:str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    candidate_k: int = 10
    rerank_k:int = 3