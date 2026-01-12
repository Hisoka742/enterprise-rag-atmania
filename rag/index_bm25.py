from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


class BM25Index:
    def __init__(self, corpus_texts: List[str]):
        self.corpus_texts = corpus_texts
        tokenized = [_tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> List[int]:
        qtok = _tokenize(query)
        scores = self.bm25.get_scores(qtok)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return ranked[: max(1, min(top_k, len(ranked)))]



