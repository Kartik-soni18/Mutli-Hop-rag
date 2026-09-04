from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from .config import Settings
from .retrieval import MetadataFilters, create_retriever


class RAGService:
    def __init__(self) -> None:
        self.settings = Settings()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model
        )
        self.vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            path=str(self.settings.qdrant_path),
            collection_name=self.settings.collection_name,
        )

    def retrieve(
        self,
        query: str,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[Document]:
        retriever = create_retriever(
            vectorstore=self.vectorstore,
            top_k=self.settings.candidate_k,
            metadata_filters=metadata_filters,
        )
        return retriever.invoke(query.strip())
