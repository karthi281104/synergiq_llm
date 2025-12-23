import os
import re
import time
import tempfile
import warnings
import hashlib
from collections import Counter

import cv2
import numpy as np
import PyPDF2
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from gtts import gTTS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_cohere.chat_models import ChatCohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_cohere.embeddings import CohereEmbeddings

try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

_AUDIO_CACHE_VERSION = 2
_VIDEO_CACHE_VERSION = 2

try:
    import asyncio
    import edge_tts

    _EDGE_TTS_AVAILABLE = True
except Exception:
    _EDGE_TTS_AVAILABLE = False

try:
    from cohere.core.api_error import ApiError as CohereApiError
except Exception:  # pragma: no cover
    CohereApiError = Exception


load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
embed_model = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
tts_voice = os.getenv("TTS_VOICE", "en-IN-PrabhatNeural")
tts_rate = os.getenv("TTS_RATE", "+0%")
tts_pitch = os.getenv("TTS_PITCH", "+0Hz")
tts_casual = os.getenv("TTS_CASUAL", "1").strip().lower() in {"1", "true", "yes", "y"}

if not api_key:
    raise RuntimeError(
        "Missing COHERE_API_KEY. Set it as an environment variable (recommended on Render) or in a .env file."
    )


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()


def clean_extracted_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines()]
    cleaned_lines: list[str] = []

    watermark_patterns = [
        re.compile(r"^www\.nammakalvi\.in\s*$", re.IGNORECASE),
        re.compile(r"^nammakalvi\s*$", re.IGNORECASE),
        re.compile(r"^namma\s+kalvi\s*$", re.IGNORECASE),
    ]

    for ln in lines:
        if not ln:
            continue
        if any(p.match(ln) for p in watermark_patterns):
            continue
        cleaned_lines.append(ln)

    deduped: list[str] = []
    prev = None
    for ln in cleaned_lines:
        if ln != prev:
            deduped.append(ln)
        prev = ln

    return "\n".join(deduped).strip()


def is_low_information_text(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True

    counts = Counter(lines)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count / max(len(lines), 1) >= 0.6:
        return True

    urlish = re.compile(r"^(https?://)?[\w.-]+\.[a-z]{2,}(/\S*)?$", re.IGNORECASE)
    urlish_lines = sum(1 for ln in lines if urlish.match(ln))
    if urlish_lines / len(lines) >= 0.6:
        return True

    tokens = re.findall(r"[A-Za-z]{3,}", text)
    if len(tokens) < 50:
        return True

    return False


def _cohere_invoke_with_retry(llm: ChatCohere, prompt: str, retries: int = 3, base_delay_s: float = 2.0) -> str:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return llm.invoke(prompt).content
        except CohereApiError as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status in (502, 503, 504) and attempt < retries - 1:
                time.sleep(base_delay_s * (2**attempt))
                continue
            raise
        except Exception as e:
            last_err = e
            raise

    if last_err:
        raise last_err
    return ""


def _prepare_text_for_tts(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bwww\.[^\s]+", " ", text)
    text = re.sub(r"\b\S+@\S+\b", " ", text)

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*(?:[-*•]+|\d+\.|\d+\))\s+", "", line)

        line = line.replace("#", " ")
        line = line.replace("•", " ")
        line = re.sub(r"\*{1,3}", "", line)

        line = re.sub(r"\s+", " ", line).strip()
        if line:
            if line[-1] not in ".!?":
                line += "."
            lines.append(line)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_text_into_slides(text: str, max_slides: int = 12, bullets_per_slide: int = 5) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    candidates: list[str] = []
    for ln in lines:
        ln = re.sub(r"^\s*(?:[-*•]+|\d+\.|\d+\))\s+", "", ln)
        ln = re.sub(r"\s+", " ", ln).strip()
        if len(ln) < 10:
            continue
        candidates.append(ln)

    expanded: list[str] = []
    for item in candidates:
        if len(item) > 180:
            parts = re.split(r"(?<=[.!?])\s+", item)
            for p in parts:
                p = p.strip()
                if len(p) >= 12:
                    expanded.append(p)
        else:
            expanded.append(item)

    bullets = [b for b in expanded if b]
    if not bullets:
        return []

    slides: list[dict] = []
    idx = 0
    while idx < len(bullets) and len(slides) < max_slides:
        chunk = bullets[idx : idx + bullets_per_slide]
        if not chunk:
            break
        title_words = re.findall(r"[A-Za-z0-9']+", chunk[0])[:6]
        title = " ".join(title_words) if title_words else f"Slide {len(slides) + 1}"
        slides.append({"title": title, "bullets": chunk})
        idx += bullets_per_slide

    return slides


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    words = (text or "").split()
    if not words:
        return ""
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        candidate = (" ".join(current + [w])).strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] <= max_width:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
            if len(lines) >= 4:
                break
    if current and len(lines) < 5:
        lines.append(" ".join(current))
    return "\n".join(lines)


def text_to_audio(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None

    if len(text) > 12000:
        text = text[:12000]

    if tts_casual:
        try:
            narrator = ChatCohere(cohere_api_key=api_key)
            rewrite_prompt = (
                "Rewrite the following content into a friendly, casual teacher-style narration in English. "
                "Keep it accurate, simple, and easy to understand. Use light conversational phrasing, but avoid rude, offensive, or unsafe slang. "
                "Do not add new facts.\n\nCONTENT:\n" + text
            )
            text = _cohere_invoke_with_retry(narrator, rewrite_prompt, retries=2, base_delay_s=1.0) or text
        except Exception:
            pass

    text = _prepare_text_for_tts(text)

    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

    if _EDGE_TTS_AVAILABLE:
        try:

            async def _run() -> None:
                communicate = edge_tts.Communicate(text=text, voice=tts_voice, rate=tts_rate, pitch=tts_pitch)
                await communicate.save(temp_audio_file.name)

            asyncio.run(_run())
            return temp_audio_file.name
        except Exception:
            pass

    if len(text) > 5000:
        text = text[:5000]
    tts = gTTS(text, lang="en")
    tts.save(temp_audio_file.name)
    return temp_audio_file.name


def text_to_animated_video(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None

    slides = _split_text_into_slides(text)
    if not slides:
        return None

    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    width, height = 1280, 720
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_file.name, fourcc, fps, (width, height))

    try:
        font_path = r"C:\\Windows\\Fonts\\Arial.ttf"
        font = ImageFont.truetype(font_path, 40)
    except IOError:
        font = ImageFont.load_default()

    title_font = font
    try:
        font_path = r"C:\\Windows\\Fonts\\Arial.ttf"
        title_font = ImageFont.truetype(font_path, 54)
        bullet_font = ImageFont.truetype(font_path, 34)
    except IOError:
        bullet_font = ImageFont.load_default()

    draw_probe = ImageDraw.Draw(Image.new("RGBA", (width, height), (0, 0, 0, 255)))
    max_text_width = width - 120

    frames_per_bullet = int(fps * 1.2)
    frames_pause = int(fps * 0.6)

    for slide_idx, slide in enumerate(slides, start=1):
        title = f"{slide_idx}. {slide['title']}"
        bullets = slide.get("bullets", [])

        wrapped_bullets: list[str] = []
        for b in bullets:
            wrapped = _wrap_text_to_width(draw_probe, b, bullet_font, max_text_width)
            wrapped_bullets.append(wrapped)

        for active_idx in range(len(wrapped_bullets)):
            for f in range(frames_per_bullet):
                alpha = int(255 * (f + 1) / max(1, frames_per_bullet))
                img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
                draw = ImageDraw.Draw(img)

                draw.text((60, 40), title, font=title_font, fill=(255, 255, 255, 255))
                draw.line((60, 110, width - 60, 110), fill=(255, 255, 255, 120), width=2)

                y = 150
                for i, b in enumerate(wrapped_bullets):
                    prefix = "• "
                    block = prefix + b
                    block = _wrap_text_to_width(draw, block, bullet_font, max_text_width)
                    bbox = draw.multiline_textbbox((0, 0), block, font=bullet_font, spacing=10)
                    block_h = bbox[3] - bbox[1]

                    if i < active_idx:
                        draw.multiline_text((80, y), block, font=bullet_font, fill=(255, 255, 255, 255), spacing=10)
                    elif i == active_idx:
                        pad_y = 10
                        draw.rectangle((70, y - 6, width - 70, y + block_h + pad_y), fill=(255, 255, 255, 20))
                        draw.multiline_text((80, y), block, font=bullet_font, fill=(255, 255, 255, alpha), spacing=10)

                    y += block_h + 30
                    if y > height - 60:
                        break

                frame = np.array(img.convert("RGB"))
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            for _ in range(frames_pause):
                img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
                draw = ImageDraw.Draw(img)
                draw.text((60, 40), title, font=title_font, fill=(255, 255, 255, 255))
                draw.line((60, 110, width - 60, 110), fill=(255, 255, 255, 120), width=2)
                y = 150
                for i, b in enumerate(wrapped_bullets):
                    if i > active_idx:
                        break
                    block = _wrap_text_to_width(draw, "• " + b, bullet_font, max_text_width)
                    bbox = draw.multiline_textbbox((0, 0), block, font=bullet_font, spacing=10)
                    block_h = bbox[3] - bbox[1]
                    draw.multiline_text((80, y), block, font=bullet_font, fill=(255, 255, 255, 255), spacing=10)
                    y += block_h + 30
                    if y > height - 60:
                        break
                frame = np.array(img.convert("RGB"))
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        for _ in range(int(fps * 0.8)):
            img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(img)
            draw.text((60, 40), title, font=title_font, fill=(255, 255, 255, 255))
            draw.line((60, 110, width - 60, 110), fill=(255, 255, 255, 120), width=2)
            y = 150
            for b in wrapped_bullets:
                block = _wrap_text_to_width(draw, "• " + b, bullet_font, max_text_width)
                bbox = draw.multiline_textbbox((0, 0), block, font=bullet_font, spacing=10)
                block_h = bbox[3] - bbox[1]
                draw.multiline_text((80, y), block, font=bullet_font, fill=(255, 255, 255, 255), spacing=10)
                y += block_h + 30
                if y > height - 60:
                    break
            frame = np.array(img.convert("RGB"))
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    return temp_video_file.name


def summarize_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "I am sorry, this information is not available."

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    chunks = splitter.split_text(text)

    if not chunks:
        return "I am sorry, this information is not available."

    summarizer = ChatCohere(cohere_api_key=api_key)
    chunk_summaries = []
    try:
        for chunk in chunks:
            prompt = (
                "Please summarize the following text into bullet points, including all key details. "
                "Ensure that the summary is detailed enough so that when all chunk summaries are combined, "
                "the final summary will span about 4 to 5 pages:\n\n" + chunk
            )
            summary_chunk = _cohere_invoke_with_retry(summarizer, prompt)
            chunk_summaries.append(summary_chunk)
    except CohereApiError as e:
        status = getattr(e, "status_code", None)
        if status in (502, 503, 504):
            return "Cohere is temporarily unavailable (server error). Please try again in 30–60 seconds."
        return "Cohere request failed. Please try again later."

    combined_summary = "\n".join(chunk_summaries)
    final_prompt = (
        "Based on the following bullet point summaries, create a comprehensive final summary "
        "that is detailed and spans about 4 to 5 pages, covering all key details:\n\n" + combined_summary
    )
    try:
        final_summary = _cohere_invoke_with_retry(summarizer, final_prompt)
        return final_summary
    except CohereApiError as e:
        status = getattr(e, "status_code", None)
        if status in (502, 503, 504):
            return "Cohere is temporarily unavailable (server error). Please try again in 30–60 seconds."
        return "Cohere request failed. Please try again later."


def build_retriever(pdf_text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = text_splitter.split_text(pdf_text)
    embeddings = CohereEmbeddings(model=embed_model, cohere_api_key=api_key, user_agent="convert.py")
    db = FAISS.from_texts(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return db, retriever


def build_chat_chain(retriever):
    template = (
        "You are a job recruiter agent working for a job portal, assigned with the task of providing "
        "information on any queries of the job seeker.\n"
        "INSTRUCTIONS:\n"
        "- Always use a friendly tone in your conversation.\n"
        "- Provide responses with bullet points, paragraphs, and proper headings when necessary.\n"
        "- Always respond in English.\n"
        "- If no relevant information is available, reply with 'I am sorry, this information is not available.'"
    )
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = ChatCohere(cohere_api_key=api_key)
    memory = ConversationBufferMemory(output_key="answer", memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return chain


def chat_answer(chain, history: list, query: str) -> str:
    try:
        result = chain({"question": query, "chat_history": history})
        history.append((query, result["answer"]))
        return result["answer"]
    except CohereApiError as e:
        status = getattr(e, "status_code", None)
        if status in (502, 503, 504):
            return "Cohere is temporarily unavailable (server error). Please try again in 30–60 seconds."
        return "Cohere request failed. Please try again later."


def file_id_from_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()
