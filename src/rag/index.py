import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import TextSplitter

from .config import Settings


def load_corpus(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as corpus_file:
        records = json.load(corpus_file)
    return [
        Document(
            page_content=item["body"],
            metadata={
                "doc_id": position,
                "title": item["title"],
                "author": item["author"],
                "source": item["source"],
                "published_at": date.fromisoformat(
                    item["published_at"][:10]
                ).isoformat(),
                "published_date_ordinal": date.fromisoformat(
                    item["published_at"][:10]
                ).toordinal(),
                "url": item["url"],
            },
        )
        for position, item in enumerate(records)
    ]


class SemanticTextSplitter(TextSplitter):
    """Group paragraphs by meaning, then enforce token budgets without overlap."""

    def __init__(
        self,
        embeddings: Embeddings,
        min_tokens: int = 100,
        target_tokens: int = 300,
        max_tokens: int = 500,
        semantic_percentile: float = 75,
    ) -> None:
        if not 0 < min_tokens <= target_tokens <= max_tokens:
            raise ValueError("Require 0 < min_tokens <= target_tokens <= max_tokens")
        if not 0 <= semantic_percentile <= 100:
            raise ValueError("semantic_percentile must be between 0 and 100")
        self.embeddings = embeddings
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.semantic_percentile = semantic_percentile
        super().__init__(
            chunk_size=max_tokens, chunk_overlap=0, length_function=self._count_tokens
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def _vectors(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self.embeddings.embed_documents(texts), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, np.finfo(float).eps)

    def _pack(self, parts: list[str], separator: str) -> list[str]:
        chunks = []
        current = ""
        for part in parts:
            candidate = separator.join(filter(None, [current, part]))
            if self._count_tokens(candidate) <= self.target_tokens:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            if self._count_tokens(part) <= self._chunk_size:
                current = part
            elif separator == "\n\n":
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z“"])', part)
                chunks.extend(self._pack(sentences, " "))
            else:
                # Only an oversized individual sentence needs token slicing.
                tokens = self.tokenizer.encode(part, disallowed_special=())
                start = 0
                while start < len(tokens):
                    end = min(start + self.target_tokens, len(tokens))
                    # Avoid cutting through a multi-token Unicode character.
                    while True:
                        try:
                            piece = self.tokenizer.decode(
                                tokens[start:end], errors="strict"
                            ).strip()
                            break
                        except UnicodeDecodeError:
                            end -= 1
                            if end == start:
                                raise ValueError(
                                    "Token target cannot fit a Unicode character"
                                ) from None
                    if piece:
                        chunks.append(piece)
                    start = end
        if current:
            chunks.append(current)
        return chunks

    def _merge_small(self, chunks: list[str]) -> list[str]:
        while len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                if self._count_tokens(chunk) >= self.min_tokens:
                    continue
                neighbors = [
                    j
                    for j in (i - 1, i + 1)
                    if 0 <= j < len(chunks)
                    and self._count_tokens(
                        "\n\n".join(chunks[min(i, j) : max(i, j) + 1])
                    )
                    <= self._chunk_size
                ]
                if not neighbors:
                    continue
                if len(neighbors) == 2:
                    vectors = self._vectors([chunk] + [chunks[j] for j in neighbors])
                    neighbor = neighbors[int(np.argmax(vectors[1:] @ vectors[0]))]
                else:
                    neighbor = neighbors[0]
                start, end = sorted((i, neighbor))
                chunks[start : end + 1] = ["\n\n".join(chunks[start : end + 1])]
                break
            else:
                break
        return chunks

    def split_text(self, text: str) -> list[str]:
        blocks = [
            cleaned
            for block in text.split("\n\n")
            if (cleaned := re.sub(r"[ \t]+", " ", block).strip())
        ]
        if not blocks:
            return []
        boundaries = []
        if len(blocks) > 1:
            vectors = self._vectors(blocks)
            distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
            threshold = np.percentile(distances, self.semantic_percentile)
            boundaries = (np.flatnonzero(distances >= threshold) + 1).tolist()
        chunks = []
        start = 0
        for end in [*boundaries, len(blocks)]:
            group = blocks[start:end]
            content = "\n\n".join(group)
            if self._count_tokens(content) <= self._chunk_size:
                chunks.append(content)
            else:
                chunks.extend(self._pack(group, "\n\n"))
            start = end
        return self._merge_small(chunks)


def split_documents(
    documents: list[Document], embeddings: Embeddings, settings: Settings
) -> list[Document]:
    splitter = SemanticTextSplitter(
        embeddings=embeddings,
        min_tokens=settings.chunk_min_tokens,
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        semantic_percentile=settings.chunk_semantic_percentile,
    )
    return splitter.split_documents(documents)


def build_index() -> int:
    settings = Settings()
    documents = load_corpus(settings.corpus_path)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    chunks = split_documents(documents, embeddings, settings)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=str(settings.qdrant_path),
        collection_name=settings.collection_name,
        force_recreate=True,
    )
    vectorstore.client.close()
    return len(chunks)


def main() -> None:
    count = build_index()
    print(f"Indexed {count} chunks.")


if __name__ == "__main__":
    main()
