import argparse
import json
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from backend.conv_logic import (
    clean_extracted_text,
    extract_pages_from_pdf,
)

try:
    from backend.conv_logic import build_retriever
except Exception:  # pragma: no cover
    build_retriever = None  # type: ignore

try:
    from langchain_cohere.chat_models import ChatCohere
except Exception:  # pragma: no cover
    ChatCohere = None  # type: ignore


NOT_FOUND = "Not found in the document."


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_BACKEND_ENV = os.path.join(_REPO_ROOT, "backend", ".env")
if os.path.exists(_BACKEND_ENV):
    load_dotenv(dotenv_path=_BACKEND_ENV)
else:
    load_dotenv()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@dataclass
class _Index:
    db: Any


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_WORD_RE.findall((s or "").lower()))


def _chunk_pages(pages: list[dict[str, Any]], *, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    # Minimal chunker to avoid depending on vector DBs/embeddings.
    # Keeps (page, chunk_id, text).
    chunks: list[dict[str, Any]] = []
    chunk_id = 0
    for p in pages:
        page_num = p.get("page")
        text = (p.get("text") or "").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(len(text), start + int(chunk_size))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"page": page_num, "chunk_id": chunk_id, "text": chunk_text})
                chunk_id += 1
            if end >= len(text):
                break
            start = max(0, end - int(chunk_overlap))
    return chunks


def _lexical_topk(question: str, chunks: list[dict[str, Any]], *, k: int) -> list[dict[str, Any]]:
    qt = _tokens(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for c in chunks:
        ct = _tokens(c.get("text") or "")
        if not ct:
            continue
        inter = len(qt & ct)
        denom = (len(qt) ** 0.5) * (len(ct) ** 0.5)
        score = (inter / denom) if denom else 0.0
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[: max(1, int(k))] if s > 0] or [c for _, c in scored[: max(1, int(k))]]


def _ensure_pdf_exists(pdf_path: str) -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")


def _build_index(pdf_path: str, doc_id: str, *, chunk_size: int, chunk_overlap: int):
    pages = extract_pages_from_pdf(pdf_path)
    for p in pages:
        p["text"] = clean_extracted_text(p.get("text") or "")

    if build_retriever is None:
        raise RuntimeError("Vector mode unavailable: build_retriever import failed")

    pdf_text = "\n".join((p.get("text") or "") for p in pages)
    db, _retriever = build_retriever(
        pdf_text,
        doc_id=doc_id,
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        search_type="similarity",
        k=5,
        fetch_k=20,
    )
    return _Index(db=db)


def _format_evidence(doc: Any, *, max_chars: int) -> str:
    text = (getattr(doc, "page_content", "") or "").strip()
    md = getattr(doc, "metadata", {}) or {}
    page = md.get("page")
    chunk_id = md.get("chunk_id")
    label = f"[p{page}:c{chunk_id}]" if (page is not None and chunk_id is not None) else ""

    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"

    return (label + " " + text).strip()


def _draft_answer_with_llm(question: str, evidence: list[str], *, model: str) -> str:
    if ChatCohere is None:
        raise RuntimeError("langchain_cohere is not available; install backend requirements")

    llm = ChatCohere(model=model)

    ev_block = "\n".join(f"- {e}" for e in evidence if e.strip())
    prompt = textwrap.dedent(
        f"""
        You are helping create a gold-answer key for a PDF QA dataset.

        RULES:
        - Use ONLY the EVIDENCE lines.
        - If the answer is not present in evidence, output exactly: {NOT_FOUND}
        - Keep the answer short and factual.
        - Do NOT include citations.

        QUESTION:
        {question}

        EVIDENCE:
        {ev_block}

        GOLD ANSWER:
        """
    ).strip()

    out = llm.invoke(prompt)
    text = (getattr(out, "content", "") or "").strip()
    return text or NOT_FOUND


def _strip_label(line: str) -> str:
    line = (line or "").strip()
    line = re.sub(r"^\[p(\?|\d+):c\d+\]\s*", "", line)
    return line.strip()


def _draft_answer_offline(evidence: list[str], *, max_chars: int = 260) -> str:
    # Offline heuristic draft: extractive snippet from the top evidence.
    # Intended only for bootstrapping; must be human-reviewed.
    for e in evidence:
        e = _strip_label(e)
        if not e:
            continue
        text = " ".join(e.split())
        if not text:
            continue
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        return text
    return NOT_FOUND


def main() -> None:
    p = argparse.ArgumentParser(description="Generate draft gold_answer and gold_evidence suggestions for eval/dataset.jsonl.")
    p.add_argument("--dataset", default="eval/dataset.jsonl")
    p.add_argument("--out", default="eval/dataset.drafts.jsonl")
    p.add_argument("--max-rows", type=int, default=0, help="If >0, process only first N rows.")
    p.add_argument("--only-missing", action="store_true", help="Only add drafts for rows where gold_answer is empty.")
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--k", type=int, default=4, help="Number of evidence chunks to attach.")
    p.add_argument("--evidence-max-chars", type=int, default=350)
    p.add_argument(
        "--retrieval",
        default="lexical",
        choices=["lexical", "vector"],
        help="Evidence retrieval method. 'lexical' uses local token overlap (no API). 'vector' uses embeddings/FAISS.",
    )
    p.add_argument("--draft-with-llm", action="store_true", help="Also generate draft_gold_answer using the chat model.")
    p.add_argument(
        "--draft-offline",
        action="store_true",
        help="Generate draft_gold_answer using an offline extractive heuristic (no API).",
    )
    p.add_argument(
        "--write-to-gold",
        action="store_true",
        help="If set, copy draft fields into gold_answer/gold_evidence when gold is empty (bootstrapping; review manually).",
    )
    p.add_argument("--llm-model", default=os.getenv("COHERE_CHAT_MODEL", "command-r-08-2024"))
    args = p.parse_args()

    rows = load_jsonl(args.dataset)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    cache: dict[tuple[str, str], _Index] = {}
    lexical_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    out_rows: list[dict[str, Any]] = []
    for r in rows:
        doc_id = str(r.get("doc_id") or "").strip()
        pdf_path = str(r.get("pdf_path") or "").strip()
        question = str(r.get("question") or "").strip()
        gold = str(r.get("gold_answer") or "").strip()

        if not doc_id or not pdf_path or not question:
            raise ValueError(f"Invalid dataset row: {r}")

        if args.only_missing and gold:
            out_rows.append(r)
            continue

        _ensure_pdf_exists(pdf_path)

        key = (pdf_path, doc_id)
        evidence: list[str] = []
        if args.retrieval == "vector":
            if key not in cache:
                cache[key] = _build_index(pdf_path, doc_id, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            idx = cache[key]
            docs = idx.db.similarity_search(question, k=int(args.k))
            evidence = [_format_evidence(d, max_chars=int(args.evidence_max_chars)) for d in docs]
        else:
            # lexical mode: local-only, no embeddings.
            if key not in lexical_cache:
                pages = extract_pages_from_pdf(pdf_path)
                for p0 in pages:
                    p0["text"] = clean_extracted_text(p0.get("text") or "")
                lexical_cache[key] = _chunk_pages(pages, chunk_size=int(args.chunk_size), chunk_overlap=int(args.chunk_overlap))

            chunks = lexical_cache[key]
            top = _lexical_topk(question, chunks, k=int(args.k))
            for c in top:
                page = c.get("page")
                chunk_id = c.get("chunk_id")
                label = f"[p{page}:c{chunk_id}]" if page is not None and chunk_id is not None else ""
                txt = (c.get("text") or "").strip()
                if int(args.evidence_max_chars) > 0 and len(txt) > int(args.evidence_max_chars):
                    txt = txt[: int(args.evidence_max_chars)].rstrip() + "…"
                evidence.append((label + " " + txt).strip())

        rr = dict(r)
        draft_ev = "\n".join(evidence).strip()
        rr["draft_gold_evidence"] = draft_ev

        if args.draft_with_llm:
            rr["draft_gold_answer"] = _draft_answer_with_llm(question, evidence, model=args.llm_model)
        elif args.draft_offline:
            rr["draft_gold_answer"] = _draft_answer_offline(evidence)

        if args.write_to_gold and not gold:
            if draft_ev and not str(rr.get("gold_evidence") or "").strip():
                rr["gold_evidence"] = draft_ev
            if args.draft_with_llm or args.draft_offline:
                da = str(rr.get("draft_gold_answer") or "").strip()
                if da and not str(rr.get("gold_answer") or "").strip():
                    rr["gold_answer"] = da

        out_rows.append(rr)

    write_jsonl(args.out, out_rows)
    print(f"Wrote drafts: {args.out}")
    print(f"Rows: {len(out_rows)}")


if __name__ == "__main__":
    main()
