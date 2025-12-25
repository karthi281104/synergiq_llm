import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from dotenv import load_dotenv

from backend.conv_logic import (
    build_chat_chain,
    build_retriever,
    extract_pages_from_pdf,
    clean_extracted_text,
)

from langchain_cohere.chat_models import ChatCohere


NOT_FOUND = "Not found in the document."


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_BACKEND_ENV = os.path.join(_REPO_ROOT, "backend", ".env")
if os.path.exists(_BACKEND_ENV):
    load_dotenv(dotenv_path=_BACKEND_ENV)
else:
    load_dotenv()


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common: dict[str, int] = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    precision = overlap / max(len(p), 1)
    recall = overlap / max(len(g), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_pdf_exists(pdf_path: str) -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")


def _build_context_index(pdf_path: str, doc_id: str, *, chunk_size: int, chunk_overlap: int, search_type: str, k: int):
    pages = extract_pages_from_pdf(pdf_path)
    for p in pages:
        p["text"] = clean_extracted_text(p.get("text") or "")

    pdf_text = "\n".join((p.get("text") or "") for p in pages)
    db, retriever = build_retriever(
        pdf_text,
        doc_id=doc_id,
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        search_type=search_type,
        k=k,
        fetch_k=max(k * 4, k),
        lambda_mult=0.5 if search_type == "mmr" else None,
    )
    return db, retriever


def _llm_only_answer(question: str, *, model: str) -> str:
    llm = ChatCohere(model=model)
    prompt = (
        "Answer the question as best as you can. "
        "If you are unsure, say you are unsure.\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
    return llm.invoke(prompt).content


@dataclass
class JudgeResult:
    supported: bool
    citation_precision: float | None


def _judge_faithfulness(question: str, answer: str, sources: list[dict], *, model: str) -> JudgeResult:
    # For baselines without citations, we only judge support.
    llm = ChatCohere(model=model)

    context_blocks: list[str] = []
    for s in sources[:5]:
        md = s.get("metadata") or {}
        page = md.get("page")
        chunk_id = md.get("chunk_id")
        label = f"p{page if page is not None else '?'}:c{chunk_id}"
        context_blocks.append(f"[{label}]\n{s.get('text','')}")

    judge_prompt = (
        "You are evaluating whether an answer is supported by provided evidence chunks from a PDF.\n"
        "Return ONLY valid JSON with keys: supported (true/false), citation_precision (number or null).\n\n"
        "Rules:\n"
        "- supported=true only if every major claim in ANSWER is supported by EVIDENCE.\n"
        "- If ANSWER is exactly 'Not found in the document.', supported=true if EVIDENCE does not contain the answer.\n"
        "- citation_precision: if ANSWER contains citations like [p?:c12] or [p3:c7], estimate the fraction of citations that actually support the adjacent claim; else null.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "EVIDENCE:\n" + "\n\n".join(context_blocks)
    )

    raw = llm.invoke(judge_prompt).content
    try:
        data = json.loads(raw)
        supported = bool(data.get("supported"))
        cp = data.get("citation_precision")
        if cp is None:
            return JudgeResult(supported=supported, citation_precision=None)
        try:
            cp_f = float(cp)
        except Exception:
            cp_f = None
        return JudgeResult(supported=supported, citation_precision=cp_f)
    except Exception:
        # If judge fails to return JSON, be conservative.
        return JudgeResult(supported=False, citation_precision=None)


def run_method(
    method: str,
    *,
    doc_id: str,
    pdf_path: str,
    question: str,
    chunk_size: int,
    chunk_overlap: int,
    k: int,
    llm_model: str,
):
    if method == "llm_only":
        answer = _llm_only_answer(question, model=llm_model)
        return {"answer": answer, "sources": []}

    search_type = "similarity" if method in {"rag_similarity", "ours"} else "mmr"
    _db, retriever = _build_context_index(
        pdf_path,
        doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        search_type=search_type,
        k=k,
    )

    strict = method == "ours"
    chain = build_chat_chain(retriever, strict=strict)

    # Keep per-question history empty for evaluation reproducibility.
    result = {"answer": "", "sources": []}
    out = chain({"question": question, "chat_history": []})
    result["answer"] = out.get("answer", "")

    if strict:
        # conv_logic.chat_answer formats sources, but for eval we can format ourselves.
        src_docs = out.get("source_documents") or []
        sources: list[dict] = []
        for d in src_docs[:5]:
            md = getattr(d, "metadata", {}) or {}
            sources.append(
                {
                    "text": (getattr(d, "page_content", "") or "").strip(),
                    "metadata": {
                        "doc_id": md.get("doc_id"),
                        "page": md.get("page"),
                        "chunk_id": md.get("chunk_id"),
                    },
                }
            )
        result["sources"] = sources
    else:
        # Baselines still retrieve; keep sources for faithfulness judging.
        src_docs = out.get("source_documents") or []
        sources = []
        for d in src_docs[:5]:
            md = getattr(d, "metadata", {}) or {}
            sources.append(
                {
                    "text": (getattr(d, "page_content", "") or "").strip(),
                    "metadata": {"doc_id": md.get("doc_id"), "page": md.get("page"), "chunk_id": md.get("chunk_id")},
                }
            )
        result["sources"] = sources

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--out", default="results/results.csv")
    parser.add_argument("--llm-model", default=os.getenv("COHERE_CHAT_MODEL", "command-r-08-2024"))
    parser.add_argument("--judge-model", default=os.getenv("COHERE_JUDGE_MODEL", os.getenv("COHERE_CHAT_MODEL", "command-r-08-2024")))
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--methods",
        default="llm_only,rag_similarity,rag_mmr,ours",
        help="Comma-separated: llm_only,rag_similarity,rag_mmr,ours",
    )

    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fieldnames = [
        "method",
        "doc_id",
        "pdf_path",
        "question",
        "gold_answer",
        "answer",
        "em",
        "f1",
        "faithful",
        "citation_precision",
        "latency_s",
    ]

    all_records: list[dict[str, Any]] = []

    for r in rows:
        doc_id = str(r.get("doc_id") or "").strip()
        pdf_path = str(r.get("pdf_path") or "").strip()
        question = str(r.get("question") or "").strip()
        gold = str(r.get("gold_answer") or "").strip()

        if not doc_id or not pdf_path or not question:
            raise ValueError(f"Invalid dataset row: {r}")

        _ensure_pdf_exists(pdf_path)

        for method in methods:
            t0 = time.perf_counter()
            result = run_method(
                method,
                doc_id=doc_id,
                pdf_path=pdf_path,
                question=question,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                k=args.k,
                llm_model=args.llm_model,
            )
            latency = time.perf_counter() - t0

            answer = (result.get("answer") or "").strip()
            sources = result.get("sources") or []

            em = exact_match(answer, gold)
            f1 = token_f1(answer, gold)

            judge = _judge_faithfulness(question, answer, sources, model=args.judge_model)

            rec = {
                "method": method,
                "doc_id": doc_id,
                "pdf_path": pdf_path,
                "question": question,
                "gold_answer": gold,
                "answer": answer,
                "em": em,
                "f1": f1,
                "faithful": 1 if judge.supported else 0,
                "citation_precision": "" if judge.citation_precision is None else judge.citation_precision,
                "latency_s": round(latency, 4),
            }
            all_records.append(rec)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    # Simple aggregate summary
    summary_path = os.path.join(os.path.dirname(args.out), "summary.md")
    by_method: dict[str, list[dict[str, Any]]] = {}
    for rec in all_records:
        by_method.setdefault(rec["method"], []).append(rec)

    def _avg(vals: Iterable[float]) -> float:
        vals = list(vals)
        return sum(vals) / max(len(vals), 1)

    lines = ["# Evaluation Summary\n", "| method | EM | F1 | Faithfulness | Citation precision | Latency (s) |", "|---|---:|---:|---:|---:|---:|"]
    for m in methods:
        recs = by_method.get(m, [])
        em_avg = _avg(float(r["em"]) for r in recs)
        f1_avg = _avg(float(r["f1"]) for r in recs)
        faith_avg = _avg(float(r["faithful"]) for r in recs)
        cp_vals = [float(r["citation_precision"]) for r in recs if str(r.get("citation_precision") or "").strip() != ""]
        cp_avg = _avg(cp_vals) if cp_vals else float("nan")
        lat_avg = _avg(float(r["latency_s"]) for r in recs)
        cp_cell = f"{cp_avg:.3f}" if cp_vals else "N/A"
        lines.append(f"| {m} | {em_avg:.3f} | {f1_avg:.3f} | {faith_avg:.3f} | {cp_cell} | {lat_avg:.3f} |")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
