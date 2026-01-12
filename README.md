# Enterprise RAG Challenge — Solution

This repository contains a **Hybrid Retrieval-Augmented Generation (RAG)** solution for answering structured questions from enterprise PDF reports.

The system builds per-company indexes from PDF documents and uses a combination of **dense (FAISS)** and **sparse (BM25)** retrieval, followed by an LLM to generate final answers.


---

## Solution Overview

Pipeline:
1. Read PDF files (with OCR fallback)
2. Split text into chunks
3. Build embeddings and FAISS index
4. Build BM25 index
5. Hybrid retrieval (dense + sparse)
6. LLM-based answer generation
7. Export results to `submission.json`

---

## Project Structure

```bash

enterprise-rag-atmania/
├── data/
│ ├── pdf/ # PDF reports
│ └── questions.json # Competition questions
├── rag/ # RAG modules
├── outputs/ # Generated submissions
├── main.py # Entry point
├── requirements.txt
└── README.md

```

## Run

```bash
python main.py --use_ocr --use_two_pass --rerun_if_na --faiss_k 60 --bm25_k 60 --candidate_cap 120 --rerank_top_n 24 --top_k 10

