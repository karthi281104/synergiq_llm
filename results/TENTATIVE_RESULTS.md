# Tentative Paper Results (IEEE)

Generated on: 2025-12-25

This repo already contains a **tentative** baseline comparison table in `results/summary.md`.

## What’s included
- Baselines: `llm_only`, `rag_similarity`, `rag_mmr`
- Proposed: `ours`
- Metrics: EM/F1 (when gold is present), citation metrics, latency, and judge-based faithfulness (when enabled)

## How to reproduce (plain command)
From repo root:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --out results/results.paper.csv --only-public --methods llm_only,rag_similarity,rag_mmr,ours --require-gold
```

## Note about Cohere Trial keys
If you are using a Cohere **Trial** API key and hit `429 TooManyRequestsError`, you will not be able to regenerate results until you:
- switch to a Production key, or
- wait for quota reset.

You can still run without judge calls to reduce API usage:

```bash
python -m eval.run_eval --dataset eval/dataset.jsonl --out results/results.paper.csv --only-public --methods llm_only,rag_similarity,rag_mmr,ours --require-gold --skip-judge
```
