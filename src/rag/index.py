import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
                "published_at": item["published_at"],
                "url": item["url"],
            },
        )
        for position, item in enumerate(records)
    ]


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index() -> int:
    settings = Settings()
    documents = load_corpus(settings.corpus_path)
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=str(settings.qdrant_path),
        collection_name=settings.collection_name,
        force_recreate=True,
    )
    return len(chunks)


def main() -> None:
    count = build_index()
    print(f"Indexed {count} chunks.")


if __name__ == "__main__":
    main()
