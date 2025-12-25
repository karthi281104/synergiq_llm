# Evaluation

This folder contains an evaluation pipeline to produce paper-style tables:

- Baseline 1: `llm_only`
- Baseline 2: `rag_similarity`
- Baseline 3: `rag_mmr`
- Proposed: `ours` (RAG + refusal + citations)

## Dataset format
Edit `eval/dataset.jsonl`. One JSON object per line:

- `doc_id`: stable identifier for the PDF
- `pdf_path`: local path to the PDF
- `question`: question text
- `gold_answer`: reference answer text, or `Not found in the document.`
- `gold_evidence` (optional): short quote span

## Run
From repo root:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --out results/results.csv --k 5 --chunk-size 900
```

Outputs:
- `results/results.csv`
- `results/summary.md`

## Notes
- Faithfulness and citation precision are computed using an LLM-judge (Cohere). For a paper, include a small manual spot-check as well.
