from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    pdf_sha1: str
    page_index: int


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks



