# Synergiq LLM — Citation-Grounded PDF-QA (RAG)

This repo implements a **citation-grounded PDF Question Answering** backend using:
- Robust PDF text extraction and cleaning
- Chunked retrieval with FAISS + Cohere embeddings
- **Grounded answering with refusal** (“Not found in the document.”)
- **Evidence returned to the client** (`sources`: chunk text + `{doc_id, page, chunk_id}`)

## Method overview

```
PDF -> page text -> chunk docs (metadata: doc_id/page/chunk_id)
                -> FAISS retrieval (similarity/MMR)
                -> LLM answers using ONLY retrieved context
                -> answer + citations [pX:cY] + returned sources
```

### Scope
- Target: **text-based PDFs** (searchable text).
- Scanned/image PDFs require OCR; currently the API rejects “low information” PDFs.

## Setup

### Environment
Set:
- `COHERE_API_KEY` (required)

Optional:
- `COHERE_EMBED_MODEL` (default: `embed-multilingual-v3.0`)
- `COHERE_CHAT_MODEL` (chat + eval; default: `command-r-08-2024`)
- `COHERE_JUDGE_MODEL` (faithfulness/citation judging; default: same as `COHERE_CHAT_MODEL`)

### Install
From repo root:

```bash
pip install -r backend/requirements.txt
```

## Run the API (FastAPI)

```bash
python -m uvicorn backend.conv:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /upload_pdf` (multipart file)
- `POST /chat` → returns `{ answer, sources }`
- `GET /summary/{doc_id}`
- `GET /audio/{doc_id}`
- `GET /video/{doc_id}`

## Grounding + citations
- Retrieved chunks are injected into the model context with labels like `[p3:c7]`.
- The model is instructed to:
  - use **only** retrieved context
  - **refuse** when evidence is missing: `Not found in the document.`
  - attach citations for major claims

## Evaluation (paper-style)

See [eval/README.md](eval/README.md).

Quick run:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --out results/results.csv --k 5 --chunk-size 900
```

Outputs:
- `results/results.csv` (per-item metrics)
- `results/summary.md` (table you can paste into a paper)

### Baselines included
- `llm_only`: no retrieval
- `rag_similarity`: similarity retrieval
- `rag_mmr`: MMR retrieval
- `ours`: RAG + refusal + citations (citation-grounded)

## Reproducing paper tables
Run the script multiple times with different flags:
- Chunk size: `--chunk-size 600|900|1200`
- Retrieval k: `--k 3|5|10`
- Retrieval type: switch methods list `--methods rag_similarity,rag_mmr,ours`

Example ablation sweep (manual):

```bash
python -m eval.run_eval --k 3 --chunk-size 600
python -m eval.run_eval --k 5 --chunk-size 900
python -m eval.run_eval --k 10 --chunk-size 1200
```
