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

Recommended extra fields (for IEEE-safe sharing):
- `source_type`: `own` | `open` | `paid_private`
- `source_url`: URL for open PDFs
- `license`: license string for open PDFs (e.g. `CC-BY-4.0`)

## Public vs private PDFs
- Put **shareable PDFs** (your own notes or open-licensed) under `eval/pdfs/public/`
- Put **paid/copyrighted PDFs** under `eval/pdfs/private/` (this folder is gitignored)

For paper tables, run eval with `--only-public`.

## Fast dataset generation (lecture notes)
Put your PDFs into `eval/pdfs/public/`, then generate a dataset skeleton:

```bash
python -m eval.make_dataset --pdfs eval/pdfs/public --out eval/dataset.generated.jsonl --per-pdf 8 --source-type own
```

Mixed own + open PDFs (recommended):

1) Create a manifest skeleton:

```bash
python -m eval.make_sources_manifest --pdfs eval/pdfs/public --out eval/sources_manifest.json
```

2) Edit `eval/sources_manifest.json` and set per-PDF fields (`source_type`, `source_url`, `license`).

3) Generate the dataset using the manifest:

```bash
python -m eval.make_dataset --pdfs eval/pdfs/public --out eval/dataset.generated.jsonl --per-pdf 8 --sources-manifest eval/sources_manifest.json
```

Then:
- copy/rename `eval/dataset.generated.jsonl` → `eval/dataset.jsonl`
- fill in `gold_answer` and `gold_evidence` for each row

## Run
From repo root:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --out results/results.csv --k 5 --chunk-size 900
```

Paper-safe run (excludes `paid_private` rows):

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --out results/results.csv --only-public
```

Outputs:
- `results/results.csv`
- `results/summary.md`

Additional metrics in `results.csv` (useful for papers):
- `citation_coverage`: fraction of non-refusal answers that include at least one citation tag like `[p3:c7]`
- `citation_format_valid`: whether bracketed tags follow the required `[pX:cY]` format
- `judge_parse_ok`: whether the LLM-judge returned valid JSON (use this to report judge failure-rate)

## Notes
- Faithfulness and citation precision are computed using an LLM-judge (Cohere). For a paper, include a small manual spot-check as well.
