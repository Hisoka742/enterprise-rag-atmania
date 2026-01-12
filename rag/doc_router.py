from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi


def _tok(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split() if t]


@dataclass
class DocHit:
    doc_id: str
    score: float


class DocRouter:
    

    def __init__(self, doc_ids: List[str], head_texts: List[str]) -> None:
        assert len(doc_ids) == len(head_texts)
        self.doc_ids = doc_ids
        self.head_texts = head_texts
        self.bm25 = BM25Okapi([_tok(t) for t in head_texts])

    def route(self, company: str, top_k: int = 3) -> List[DocHit]:
        company = (company or "").strip()
        if not company:
            return []
        scores = self.bm25.get_scores(_tok(company))
        idxs = np.argsort(-scores)[:top_k].tolist()
        return [DocHit(self.doc_ids[i], float(scores[i])) for i in idxs]

