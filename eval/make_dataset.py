import argparse
import hashlib
import json
import os
from typing import Iterable


def _stable_doc_id(pdf_path: str) -> str:
    # Stable across machines as long as relative path + content remains same.
    # Uses file bytes so renames won't break doc_id if content unchanged.
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _default_questions() -> list[str]:
    # Designed for English lecture notes + skill notes.
    return [
        "What is the main topic of this document?",
        "List 5 key concepts covered in the notes.",
        "Define the most important terms introduced.",
        "Explain the core idea in simple steps.",
        "What procedure or workflow is described?",
        "List any assumptions or prerequisites mentioned.",
        "What formulas or equations are mentioned?",
        "What are the key advantages and limitations mentioned?",
        "Summarize the conclusion or takeaways.",
        "What are recommended references or resources mentioned?",
        "Is there any deadline, grading rubric, or submission instruction in the document? If not, answer 'Not found in the document.'",
    ]


def _iter_pdfs(folder: str) -> Iterable[str]:
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".pdf"):
                yield os.path.join(root, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval/dataset.jsonl skeleton from PDFs.")
    parser.add_argument("--pdfs", default="eval/pdfs", help="Folder containing PDFs")
    parser.add_argument("--out", default="eval/dataset.generated.jsonl", help="Output JSONL path")
    parser.add_argument("--per-pdf", type=int, default=8, help="Number of questions per PDF (max 11)")
    args = parser.parse_args()

    pdf_folder = args.pdfs
    out_path = args.out
    per_pdf = max(1, min(int(args.per_pdf), len(_default_questions())))

    pdfs = sorted(list(_iter_pdfs(pdf_folder)))
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {pdf_folder}")

    questions = _default_questions()[:per_pdf]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pdf_path in pdfs:
            doc_id = _stable_doc_id(pdf_path)
            # Use repo-relative path when possible (prettier diffs + portability)
            rel_path = os.path.relpath(pdf_path, start=os.getcwd()).replace("\\", "/")
            for q in questions:
                row = {
                    "doc_id": doc_id,
                    "pdf_path": rel_path,
                    "question": q,
                    "gold_answer": "",
                    "gold_evidence": "",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote dataset skeleton: {out_path}")
    print(f"PDFs: {len(pdfs)} | Questions per PDF: {len(questions)} | Total rows: {len(pdfs)*len(questions)}")


if __name__ == "__main__":
    main()
