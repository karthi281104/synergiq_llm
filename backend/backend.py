"""LEGACY (not used).

This file is kept for reference only.

The supported/reproducible API used for deployment is:
- backend/conv.py
- backend/conv_logic.py

Render uses: `python -m uvicorn backend.conv:app` (see render.yaml).
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Body
from pydantic import BaseModel
import PyPDF2
import tempfile
import os
import hashlib
import cv2
import numpy as np
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_cohere.chat_models import ChatCohere
from langchain.chains import ConversationalRetrievalChain
from langchain_cohere.embeddings import CohereEmbeddings
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


# Load API Key
load_dotenv()
api_key = os.getenv('COHERE_API_KEY')

app = FastAPI()

# Store Chat History
chat_histories = {}

# Cache vector stores per PDF to avoid rebuilding every chat request
vector_store_cache = {}


class ChatRequest(BaseModel):
    pdf_text: str
    question: str

cors_origins_raw = os.getenv("CORS_ORIGINS", "*").strip()
cors_allow_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()] if cors_origins_raw != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
async def health():
    return {"success": True, "service": "synergiq-llm-api"}

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

# Function to convert text to speech (TTS)
def text_to_audio(text):
    tts = gTTS(text[:5000], lang="en")  # gTTS limit
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# Function to create a scrolling text video
def text_to_animated_video(text):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    width, height, fps, duration = 1280, 720, 30, 10
    out = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    words = text.split()
    lines = [" ".join(words[i:i+10]) for i in range(0, min(len(words), 100), 10)]
    full_text = "\n".join(lines)

    for i in range(fps * duration):
        img = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(img)
        y_position = height - int(i * (height / (fps * duration)))
        draw.text((50, y_position), full_text, font=font, fill="white")
        frame = np.array(img)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    return temp_video.name

# Function to summarize PDF content
def summarize_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    chunks = splitter.split_text(text)

    summarizer = ChatCohere(model="command-r", cohere_api_key=api_key)

    chunk_summaries: list[str] = []
    for chunk in chunks:
        resp = summarizer.invoke(f"Summarize this text:\n\n{chunk}")
        chunk_summaries.append(getattr(resp, "content", str(resp)))

    resp = summarizer.invoke(
        "Combine these summaries into a 4-5 page detailed summary:\n\n" + "\n\n".join(chunk_summaries)
    )
    return getattr(resp, "content", str(resp))

# ----------------------
# API ENDPOINTS
# ----------------------

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(await file.read())
    temp_pdf.close()

    pdf_text = extract_text_from_pdf(temp_pdf.name)
    filename = os.path.basename(temp_pdf.name)  # Extract filename

    return {"text": pdf_text, "filename": filename}  # Ensure 'filename' is returned


@app.post("/summarize/")
async def summarize(payload: dict = Body(...)):
    pdf_text = payload.get("pdf_text", "")
    if not pdf_text:
        return JSONResponse(status_code=400, content={"error": "Invalid PDF text"})
    summary = summarize_text(pdf_text)
    return {"summary": summary}


@app.post("/text_to_audio/")
async def text_to_audio_api(payload: dict = Body(...)):
    pdf_text = payload.get("pdf_text", "")
    if not pdf_text:
        return JSONResponse(status_code=400, content={"error": "Invalid PDF text"})
    audio_path = text_to_audio(pdf_text)
    return FileResponse(audio_path, media_type="audio/mp3", filename="pdf_audio.mp3")

@app.post("/text_to_video/")
async def text_to_video_api(payload: dict = Body(...)):
    pdf_text = payload.get("pdf_text", "")
    if not pdf_text:
        return JSONResponse(status_code=400, content={"error": "Invalid PDF text"})
    video_path = text_to_animated_video(pdf_text)
    return FileResponse(video_path, media_type="video/mp4", filename="animated_pdf_video.mp4")


@app.post("/chat/")
async def chat_with_pdf(payload: ChatRequest):
    pdf_text = (payload.pdf_text or "").strip()
    question = (payload.question or "").strip()

    if not pdf_text or not question:
        return JSONResponse(status_code=400, content={"error": "pdf_text and question are required"})

    pdf_id = hashlib.sha256(pdf_text.encode("utf-8")).hexdigest()

    if pdf_id not in chat_histories:
        chat_histories[pdf_id] = []

    if pdf_id not in vector_store_cache:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        chunks = text_splitter.split_text(pdf_text)
        embeddings = CohereEmbeddings(cohere_api_key=api_key)
        db = FAISS.from_texts(chunks, embeddings)
        vector_store_cache[pdf_id] = db

    db = vector_store_cache[pdf_id]
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant answering queries about the PDF.\n\nQuestion: {question}"
    )
    llm = ChatCohere(cohere_api_key=api_key, model="command-r")
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

    result = chain({"question": question, "chat_history": chat_histories[pdf_id]})
    chat_histories[pdf_id].append((question, result["answer"]))

    return {"answer": result["answer"]}

# Run the FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
