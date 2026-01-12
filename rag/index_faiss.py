from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import faiss


@dataclass
class SearchResult:
    idx: int
    score: float


class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(vectors)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        scores, idxs = self.index.search(query_vec, top_k)
        res: List[SearchResult] = []
        for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
            if i == -1:
                continue
            res.append(SearchResult(idx=i, score=float(s)))
        return res

