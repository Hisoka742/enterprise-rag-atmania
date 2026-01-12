# Enterprise RAG Challenge 

Hybrid Retrieval-Augmented Generation (RAG) system for answering structured questions from enterprise PDF documents. 
**Status:** Valid submission, stable pipeline

---

## Method
- PDF text extraction (+ OCR fallback)
- Chunking
- Dense retrieval (FAISS embeddings)
- Sparse retrieval (BM25)
- Hybrid ranking
- LLM answer generation
- Strict output normalization

Indexes are built **per company** to reduce cross-document noise.

---

## Run

```bash
python main.py --use_ocr --use_two_pass --rerun_if_na --faiss_k 60 --bm25_k 60 --candidate_cap 120 --rerank_top_n 24 --top_k 10

