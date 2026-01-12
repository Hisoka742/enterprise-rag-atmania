from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import fitz  
from PIL import Image
import pytesseract


@dataclass
class PageText:
    pdf_path: str
    pdf_sha1: str
    page_index: int
    text: str


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text_pymupdf(pdf_path: str, page_index: int) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    txt = page.get_text("text") or ""
    doc.close()
    return txt.strip()


def ocr_page_pymupdf(pdf_path: str, page_index: int, dpi: int = 200) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img).strip()


def read_pdf_pages(
    pdf_path: str,
    ocr_fallback: bool = True,
    min_chars_before_ocr: int = 50,
) -> List[PageText]:
    pdf_sha1 = sha1_file(pdf_path)

    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    doc.close()

    out: List[PageText] = []
    for i in range(n_pages):
        text = extract_text_pymupdf(pdf_path, i)

        if ocr_fallback and len(text) < min_chars_before_ocr:
            try:
                ocr_text = ocr_page_pymupdf(pdf_path, i)
                if len(ocr_text) > len(text):
                    text = ocr_text
            except Exception:
                pass

        out.append(PageText(pdf_path=pdf_path, pdf_sha1=pdf_sha1, page_index=i, text=text))
    return out
