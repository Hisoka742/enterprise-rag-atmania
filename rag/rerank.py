from __future__ import annotations

from typing import List, Tuple
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder reranker for (query, passage) pairs."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, passages: List[str], top_n: int = 8) -> List[int]:
        if not passages:
            return []
        pairs: List[Tuple[str, str]] = [(query, p) for p in passages]
        scores = self.model.predict(pairs)
        ranked = sorted(range(len(passages)), key=lambda i: float(scores[i]), reverse=True)
        return ranked[: max(1, min(top_n, len(ranked)))]
