from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

from dotenv import load_dotenv

from rag.pdf_reader import read_pdf_pages
from rag.chunking import Chunk, chunk_text
from rag.embeddings import Embedder
from rag.index_faiss import FaissIndex
from rag.index_bm25 import BM25Index
from rag.rerank import Reranker
from rag.generator_openai import OpenAIGenerator
from rag.submission import make_reference, write_submission


NA = "N/A"


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    raise ValueError("questions.json must be a list or {'questions': [...]}.")
    

def build_corpus(pdf_dir: str, use_ocr: bool) -> List[Chunk]:
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    chunks: List[Chunk] = []
    for pdf_path in pdf_paths:
        pages = read_pdf_pages(pdf_path, ocr_fallback=use_ocr)
        for p in pages:
            for t in chunk_text(p.text):
                chunks.append(Chunk(text=t, pdf_sha1=p.pdf_sha1, page_index=p.page_index))
    return chunks


def normalize_boolean(raw: str) -> bool:
    t = (raw or "").strip().lower()
    return t in {"true", "yes", "y", "1"}


def normalize_number(raw: str) -> Union[float, str]:
    """
    Robust number parsing:
    - thousands: 1,234 / 1 234 / 1.234
    - euro decimals: 12,5
    - negatives: (123)
    - suffixes: k/thousand, m/million, bn/billion
    """
    t = (raw or "").strip()
    if not t or t.lower() in {"n/a", "na", "unknown", "not available"}:
        return NA

    neg = "(" in t and ")" in t
    low = t.lower()

    mult = 1.0
    if re.search(r"\b(bn|billion)\b", low):
        mult = 1e9
    elif re.search(r"\b(m|million)\b", low):
        mult = 1e6
    elif re.search(r"\b(k|thousand)\b", low):
        mult = 1e3

    m = re.search(r"-?\(?\d[\d\s,\.]*\)?", t)
    if not m:
        return NA

    num_str = m.group(0).replace("(", "").replace(")", "").strip()

    # Both '.' and ',' -> commas are thousands
    if "." in num_str and "," in num_str:
        num_str = num_str.replace(",", "")
    else:
        # Only ',' -> decimal comma if last group len 1-2, else thousands
        if "," in num_str and "." not in num_str:
            parts = num_str.split(",")
            if len(parts[-1]) in (1, 2):
                num_str = ".".join(parts)
            else:
                num_str = "".join(parts)

        # Only '.' -> if looks like thousands (1.234.567) then remove dots
        if "." in num_str and "," not in num_str:
            parts = num_str.split(".")
            if len(parts) > 1 and len(parts[-1]) == 3:
                num_str = "".join(parts)

    num_str = num_str.replace(" ", "")

    try:
        val = float(num_str) * mult
        if neg:
            val = -abs(val)
        return float(val)
    except Exception:
        return NA


def normalize_name(raw: str) -> str:
    t = (raw or "").strip()
    if not t or t.lower() in {"n/a", "na", "unknown", "not available"}:
        return NA
    return t


def normalize_names(raw: str) -> Union[List[str], str]:
    t = (raw or "").strip()
    if not t or t.lower() in {"n/a", "na", "unknown", "not available"}:
        return NA
    parts = [p.strip() for p in re.split(r"[;\n,]+", t) if p.strip()]
    return parts if parts else NA


def is_hard_question(qtext: str, kind: str) -> bool:
    t = (qtext or "").lower()
    if "which of the companies" in t or "lowest" in t:
        return True
    if "executive compensation" in t:
        return True
    if "leadership positions changed" in t:
        return True
    if any(k in t for k in [
        "cash flow from operations",
        "total revenue",
        "total assets",
        "net income",
        "operating margin",
        "gross margin",
        "capital expenditures",
        "dividend per share",
    ]):
        return True
    if kind == "names":
        return True
    return False


def should_rerun_on_fallback(value: Any, kind: str) -> bool:
    if value == NA and kind in {"number", "name", "names"}:
        return True
    return False


def _merge_candidate_ids(faiss_ids: List[int], bm25_ids: List[int], cap: int) -> List[int]:
    seen: Set[int] = set()
    merged: List[int] = []
    for idx in faiss_ids + bm25_ids:
        if idx not in seen:
            merged.append(idx)
            seen.add(idx)
        if len(merged) >= cap:
            break
    return merged


def main():
    # robust .env loading
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/pdf")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--team_email", default="atmaniaslmane@gmail.com")
    parser.add_argument("--surname", default="Atmania")
    parser.add_argument("--version", default="v0")

    parser.add_argument("--use_ocr", action="store_true")
    parser.add_argument("--embedding_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--questions_json", default="data/questions.json")

    # hybrid retrieval
    parser.add_argument("--faiss_k", type=int, default=40, help="FAISS candidates")
    parser.add_argument("--bm25_k", type=int, default=40, help="BM25 candidates")
    parser.add_argument("--candidate_cap", type=int, default=80, help="max unique candidates before rerank")

    # rerank + context
    parser.add_argument("--rerank_top_n", type=int, default=16, help="how many to keep after rerank")
    parser.add_argument("--top_k", type=int, default=8, help="final chunks used in context (<= rerank_top_n)")
    parser.add_argument("--reranker_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # two-pass LLM
    parser.add_argument("--use_two_pass", action="store_true")
    parser.add_argument("--rerun_if_na", action="store_true")
    parser.add_argument("--primary_model", default=None)
    parser.add_argument("--fallback_model", default=None)

    args = parser.parse_args()

    primary_model = os.getenv("OPENAI_MODEL_PRIMARY", "gpt-5-mini")
    fallback_model = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-5.2")
    if args.primary_model:
        primary_model = args.primary_model
    if args.fallback_model:
        fallback_model = args.fallback_model

    submission_name = f"{args.surname}_{args.version}"
    out_path = os.path.join(args.out_dir, f"submission_{submission_name}.json")

    print("Building corpus...")
    chunks = build_corpus(args.data_dir, use_ocr=args.use_ocr)
    print(f"Chunks: {len(chunks)}")

    texts = [c.text for c in chunks]

    print("Building indexes (FAISS + BM25)...")
    embedder = Embedder(args.embedding_model)
    vecs = embedder.encode(texts)

    faiss_index = FaissIndex(dim=vecs.shape[1])
    faiss_index.add(vecs)

    bm25_index = BM25Index(texts)

    reranker = Reranker(args.reranker_model)
    gen = OpenAIGenerator(model=primary_model)

    questions = load_questions(args.questions_json)

    answers_out: List[Dict[str, Any]] = []

    for qi, item in enumerate(questions, start=1):
        qtext = str(item.get("text", "")).strip()
        kind = str(item.get("kind", "name")).strip()

        # FAISS candidates
        qvec = embedder.encode([qtext])
        faiss_hits = faiss_index.search(qvec, top_k=args.faiss_k)
        faiss_ids = [h.idx for h in faiss_hits]

        # BM25 candidates
        bm25_ids = bm25_index.search(qtext, top_k=args.bm25_k)

        # Merge unique
        cand_ids = _merge_candidate_ids(faiss_ids, bm25_ids, cap=args.candidate_cap)
        cand_chunks = [chunks[i] for i in cand_ids]
        cand_texts = [c.text for c in cand_chunks]

        # Rerank
        rerank_ids = reranker.rerank(qtext, cand_texts, top_n=args.rerank_top_n)
        rerank_ids = rerank_ids[: max(1, args.rerank_top_n)]

        # Build context from top_k reranked
        final_ids = rerank_ids[: min(args.top_k, len(rerank_ids))]

        context_parts: List[str] = []
        refs: List[Dict[str, Any]] = []
        used: Set[Tuple[str, int]] = set()

        for j in final_ids:
            c = cand_chunks[j]
            context_parts.append(c.text)
            key = (c.pdf_sha1, c.page_index)
            if len(refs) < 3 and key not in used:
                refs.append(make_reference(c.pdf_sha1, c.page_index))
                used.add(key)

        context = "\n\n---\n\n".join(context_parts)[:14000]

        # 1st pass
        raw = gen.answer(qtext, kind, context, model_override=primary_model)

        if kind == "boolean":
            value: Any = normalize_boolean(raw)
        elif kind == "number":
            value = normalize_number(raw)
        elif kind == "name":
            value = normalize_name(raw)
        elif kind == "names":
            value = normalize_names(raw)
        else:
            value = normalize_name(raw)

        references = [] if value == NA else refs

        answers_out.append(
            {
                "question_text": qtext,
                "kind": kind,
                "value": value,
                "references": references,
                "_context": context,
                "_refs": refs,
            }
        )

        if qi % 5 == 0:
            print(f"Answered {qi}/{len(questions)}")

    # 2nd pass (only hard / N/A)
    if args.use_two_pass:
        print(f"Two-pass enabled. primary={primary_model} fallback={fallback_model}")
        applied = 0

        for a in answers_out:
            qtext = a["question_text"]
            kind = a["kind"]
            value = a["value"]

            hard = is_hard_question(qtext, kind)
            rerun_na = args.rerun_if_na and should_rerun_on_fallback(value, kind)

            if not hard and not rerun_na:
                continue

            context = a.get("_context", "")
            refs = a.get("_refs", [])

            raw2 = gen.answer(qtext, kind, context, model_override=fallback_model)

            if kind == "boolean":
                value2: Any = normalize_boolean(raw2)
            elif kind == "number":
                value2 = normalize_number(raw2)
            elif kind == "name":
                value2 = normalize_name(raw2)
            elif kind == "names":
                value2 = normalize_names(raw2)
            else:
                value2 = normalize_name(raw2)

            # accept improvements safely
            if value == NA and value2 != NA:
                a["value"] = value2
                a["references"] = [] if value2 == NA else refs
                applied += 1
            elif value != NA and value2 != NA:
                a["value"] = value2
                a["references"] = [] if value2 == NA else refs
                applied += 1

        print(f"Fallback reruns applied: {applied}")

    # remove service fields
    for a in answers_out:
        a.pop("_context", None)
        a.pop("_refs", None)

    write_submission(out_path, args.team_email, submission_name, answers_out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

