import os
import tempfile
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.conv_logic import (
    extract_text_from_pdf,
    extract_pages_from_pdf,
    clean_extracted_text,
    is_low_information_text,
    summarize_text,
    text_to_audio,
    text_to_animated_video,
    build_retriever,
    build_chat_chain,
    chat_answer,
    file_id_from_bytes,
)

app = FastAPI(title="synergiq-conv-rest-api")


class ChatRequest(BaseModel):
    doc_id: str
    question: str


_DOCS: dict[str, dict[str, Any]] = {}


@app.get("/health")
def health():
    return {"ok": True, "service": "synergiq-conv-rest-api"}


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    data = await file.read()
    doc_id = file_id_from_bytes(data)

    if doc_id not in _DOCS:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(data)
            pages = extract_pages_from_pdf(temp_pdf.name)
            pdf_text = "\n".join((p.get("text") or "") for p in pages)

        pdf_text = clean_extracted_text(pdf_text)
        if is_low_information_text(pdf_text):
            raise HTTPException(
                status_code=400,
                detail=(
                    "The extracted text looks like a watermark/URL-only (or too little text). "
                    "This usually happens with scanned/image PDFs or protected PDFs. "
                    "Try an OCR'd (searchable text) PDF and upload again."
                ),
            )

        db, retriever = build_retriever(pdf_text, doc_id=doc_id, pages=pages)

        _DOCS[doc_id] = {
            "pdf_text": pdf_text,
            "vector_db": db,
            "retriever": retriever,
            "history": [],
            "summary": None,
            "audio_file_path": None,
            "video_file_path": None,
        }

    return {"doc_id": doc_id, "pdf_text": _DOCS[doc_id]["pdf_text"]}


@app.get("/text/{doc_id}")
def get_text(doc_id: str):
    doc = _DOCS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")
    return {"doc_id": doc_id, "pdf_text": doc["pdf_text"]}


@app.get("/summary/{doc_id}")
def get_summary(doc_id: str):
    doc = _DOCS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")

    if not doc.get("summary"):
        doc["summary"] = summarize_text(doc["pdf_text"])
    return {"doc_id": doc_id, "summary": doc["summary"]}


@app.get("/audio/{doc_id}")
def get_audio(doc_id: str):
    doc = _DOCS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")

    if not doc.get("audio_file_path"):
        doc["audio_file_path"] = text_to_audio(doc["pdf_text"])

    audio_file_path = doc.get("audio_file_path")
    if not audio_file_path:
        raise HTTPException(status_code=400, detail="Audio not generated because no text was available.")
    return FileResponse(audio_file_path, media_type="audio/mpeg", filename="pdf_audio.mp3")


@app.get("/video/{doc_id}")
def get_video(doc_id: str):
    doc = _DOCS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")

    if not doc.get("video_file_path"):
        doc["video_file_path"] = text_to_animated_video(doc["pdf_text"])

    video_file_path = doc.get("video_file_path")
    if not video_file_path:
        raise HTTPException(status_code=400, detail="Video not generated because no text was available.")
    return FileResponse(video_file_path, media_type="video/mp4", filename="animated_pdf_video.mp4")


@app.post("/chat")
def chat(req: ChatRequest):
    doc = _DOCS.get(req.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a PDF first.")

    retriever = doc["retriever"]
    chain = build_chat_chain(retriever, strict=True)
    result = chat_answer(chain, doc["history"], req.question)
    return {"doc_id": req.doc_id, "question": req.question, **result}