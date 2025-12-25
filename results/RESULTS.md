# Synergiq : An End-to-End LLM System for Educational Notes With Grounded Chat and Audio-Summary Generation

Generated on: 2025-12-25

This repo can generate a comparison table in `results/summary.md` and raw rows in `results/results.csv`.

## What’s included
- Baselines: `llm_only`, `rag_similarity`, `rag_mmr`
- Proposed: `ours`
- Metrics: EM/F1 (when gold is present), citation metrics, latency, and judge-based faithfulness (when enabled)

## How to reproduce (plain command)
From repo root:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --out results/results.csv --only-public --methods llm_only,rag_similarity,rag_mmr,ours --skip-judge
```

Note: `eval/dataset.jsonl` currently contains **bootstrapped (auto-filled) gold_answer/gold_evidence** generated offline from retrieved snippets. This is meant to accelerate labeling and should be reviewed/edited for a paper.

Paper EM/F1 run (after reviewing/fixing `gold_answer`/`gold_evidence`):

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --out results/results.paper.csv --only-public --methods llm_only,rag_similarity,rag_mmr,ours --require-gold
```

## Note about Cohere Trial keys
If you are using a Cohere **Trial** API key and hit `429 TooManyRequestsError`, you may not be able to regenerate results until you:
- switch to a Production key, or
- wait for quota reset.

You can run without judge calls to reduce API usage:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --out results/results.paper.csv --only-public --methods llm_only,rag_similarity,rag_mmr,ours --require-gold --skip-judge
```
