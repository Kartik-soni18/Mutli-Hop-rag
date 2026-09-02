from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from .config import Settings


def create_retriever(
    settings: Settings,
    embeddings: HuggingFaceEmbeddings,
) -> BaseRetriever:
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=str(settings.qdrant_path),
        collection_name=settings.collection_name,
    )
    return vectorstore.as_retriever(search_kwargs={"k": settings.top_k})


def retrieve(query: str, retriever: BaseRetriever) -> list[Document]:
    return retriever.invoke(query.strip())
