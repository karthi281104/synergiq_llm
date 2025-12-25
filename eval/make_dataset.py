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
        "What is the main topic of these notes?",
        "List 5 key concepts covered.",
        "Define two important terms exactly as stated in the notes.",
        "Write one formula/equation mentioned in the notes (as written).",
        "List the steps of one algorithm/procedure described in the notes.",
        "Compare two related concepts/methods mentioned (give at least one difference).",
        "List any assumptions, prerequisites, or constraints mentioned.",
        "Give one example mentioned in the notes.",
        "What are the advantages and limitations of the main method/topic (as stated)?",
        "Summarize the key takeaways in 3 bullet points.",
        "Is there any deadline/submission instruction in the document? If not, answer 'Not found in the document.'",
        "What is one definition stated in the notes? Quote it exactly if present; otherwise answer 'Not found in the document.'",
        "What is the time complexity of one algorithm mentioned (as stated)? If not present, answer 'Not found in the document.'",
        "List 3 important keywords/terms that appear in the notes.",
        "What is one theorem/lemma/property stated? If not present, answer 'Not found in the document.'",
        "Describe one table/list mentioned (what items are compared or listed). If none, answer 'Not found in the document.'",
        "What is one stated limitation/constraint? If none, answer 'Not found in the document.'",
        "What is one stated advantage/benefit? If none, answer 'Not found in the document.'",
        "Provide one short quote (1–2 sentences) from the notes that best represents the topic.",
        "What is one abbreviation/acronym expanded in the notes? If none, answer 'Not found in the document.'",
    ]


def _iter_pdfs(folder: str) -> Iterable[str]:
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".pdf"):
                yield os.path.join(root, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval/dataset.jsonl skeleton from PDFs.")
    parser.add_argument("--pdfs", default="eval/pdfs/public", help="Folder containing PDFs")
    parser.add_argument("--out", default="eval/dataset.generated.jsonl", help="Output JSONL path")
    parser.add_argument("--per-pdf", type=int, default=8, help="Number of questions per PDF (max: size of built-in question bank)")
    parser.add_argument(
        "--sources-manifest",
        default="",
        help="Optional JSON manifest that maps each pdf_path to source_type/source_url/license (recommended for mixed datasets).",
    )
    parser.add_argument(
        "--source-type",
        default="own",
        choices=["own", "open", "paid_private"],
        help="Source category for the PDFs in --pdfs",
    )
    parser.add_argument("--source-url", default="", help="Optional URL (for open PDFs)")
    parser.add_argument("--license", default="", help="Optional license string (e.g., CC-BY-4.0)")
    args = parser.parse_args()

    pdf_folder = args.pdfs
    out_path = args.out
    per_pdf = max(1, min(int(args.per_pdf), len(_default_questions())))
    source_type = args.source_type
    source_url = (args.source_url or "").strip()
    license_str = (args.license or "").strip()

    manifest_path = (args.sources_manifest or "").strip()
    manifest: dict | None = None
    if manifest_path:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

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

            row_source_type = source_type
            row_source_url = source_url
            row_license = license_str
            if manifest:
                defaults = manifest.get("defaults", {}) if isinstance(manifest, dict) else {}
                entry = None
                pdf_map = manifest.get("pdfs", {}) if isinstance(manifest, dict) else {}
                if isinstance(pdf_map, dict):
                    entry = pdf_map.get(rel_path)

                if isinstance(defaults, dict):
                    row_source_type = str(defaults.get("source_type", row_source_type) or row_source_type)
                    row_source_url = str(defaults.get("source_url", row_source_url) or row_source_url)
                    row_license = str(defaults.get("license", row_license) or row_license)

                if isinstance(entry, dict):
                    row_source_type = str(entry.get("source_type", row_source_type) or row_source_type)
                    row_source_url = str(entry.get("source_url", row_source_url) or row_source_url)
                    row_license = str(entry.get("license", row_license) or row_license)

            for q in questions:
                row = {
                    "doc_id": doc_id,
                    "pdf_path": rel_path,
                    "source_type": row_source_type,
                    "source_url": row_source_url,
                    "license": row_license,
                    "question": q,
                    "gold_answer": "",
                    "gold_evidence": "",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote dataset skeleton: {out_path}")
    print(f"PDFs: {len(pdfs)} | Questions per PDF: {len(questions)} | Total rows: {len(pdfs)*len(questions)}")


if __name__ == "__main__":
    main()
