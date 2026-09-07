from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from .config import Settings
from .retrieval import MetadataFilters, build_metadata_filter


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
        return self.search(self.embeddings.embed_query(query.strip()), metadata_filters)

    def search(self, vector, metadata_filters=None):
        pairs = self.vectorstore.similarity_search_with_score_by_vector(
            vector,
            k=self.settings.candidate_k,
            filter=build_metadata_filter(metadata_filters),
        )
        documents = []
        for document, score in pairs:
            document = document.model_copy(deep=True)
            document.metadata["similarity_score"] = float(score)
            documents.append(document)
        return documents
