from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings
from .retrieval import create_retriever, retrieve


class RAGService:
    def __init__(self) -> None:
        self.settings = Settings()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model
        )
        self.retriever = create_retriever(self.settings, self.embeddings)

    def retrieve(self, query: str) -> list[Document]:
        return retrieve(query, self.retriever)
