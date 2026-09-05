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
            document = document.model_copy(deep=True)
            document.metadata["reranker_score"] = float(score)
            results.append(document)
        return results[:min(Settings().rerank_k, 6)]

    def rerank_branches(
        self,
        queries: list[str],
        documents: list[Document],
    ) -> list[Document]:
        """Score within branches, then share at most six unique slots fairly."""
        if not documents or not queries:
            return []
        pairs = [
            (branch_id, document_id)
            for branch_id in range(len(queries))
            for document_id, document in enumerate(documents)
            if branch_id in document.metadata.get("branch_ids", [])
        ]
        if not pairs:
            return []
        scores = self.model.predict([
            (queries[branch_id], documents[document_id].page_content)
            for branch_id, document_id in pairs
        ])
        rankings = [[] for _ in queries]
        branch_scores = [{} for _ in documents]
        for (branch_id, document_id), score in zip(pairs, scores, strict=True):
            rankings[branch_id].append((document_id, float(score)))
            branch_scores[document_id][str(branch_id)] = float(score)
        for ranking in rankings:
            ranking.sort(key=lambda item: item[1], reverse=True)

        selected = []
        selected_ids = set()
        coverage = [0] * len(queries)
        limit = min(Settings().rerank_k, 6)
        while len(selected) < limit:
            # Shared chunks occupy one slot and count toward all their branches.
            for ranking in rankings:
                while ranking and ranking[0][0] in selected_ids:
                    ranking.pop(0)
            available = [i for i, ranking in enumerate(rankings) if ranking]
            if not available:
                break
            branch_id = min(available, key=lambda i: (coverage[i], i))
            document_id, _ = rankings[branch_id].pop(0)
            selected_ids.add(document_id)
            document = documents[document_id].model_copy(deep=True)
            document.metadata["branch_reranker_scores"] = branch_scores[document_id]
            selected.append(document)
            for member in range(len(queries)):
                if str(member) in branch_scores[document_id]:
                    coverage[member] += 1
        return selected
