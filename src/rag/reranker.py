from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from .config import Settings


class DocumentReranker:
      def __init__(self, model_name: str) -> None:
          self.model = CrossEncoder(model_name)

      def rerank(
          self,
          query: str,
          documents: list[Document],
      ) -> list[Document]:
          if not documents:
              return []
          settings=Settings()
          scores = self.model.predict(
              [(query, document.page_content) for document in documents]
          )

          ranked = sorted(
              zip(documents, scores, strict=True),
              key=lambda item: float(item[1]),
              reverse=True,
          )

          results = []
          for document, score in ranked:
              document.metadata["reranker_score"] = float(score)
              results.append(document)

          return results[:settings.rerank_k]