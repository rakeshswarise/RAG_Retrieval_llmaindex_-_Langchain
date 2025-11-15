import os
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------
# LOGGING (CONSOLE + ROTATING FILE LOGGER)
# ---------------------------------------------------

if not os.path.exists("logs"):
    os.makedirs("logs")

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

# File logger (5 MB per file, keep 5 files)
file_handler = RotatingFileHandler(
    "logs/app.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# Console logger
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Main logger
log = logging.getLogger("rag-fast")
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log.addHandler(console_handler)
log.propagate = False

# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# RAG IMPORTS
# ---------------------------------------------------

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

GEMINI_API_KEY = ""

Settings.log_level = "INFO"

index = None
llm_fast = None


# ---------------------------------------------------
# MODELS
# ---------------------------------------------------

class QuizRequest(BaseModel):
    topic: str
    difficulty: str

class Answer(BaseModel):
    selected: str
    correct: str

class ScoreRequest(BaseModel):
    answers: List[Answer]


# ---------------------------------------------------
# CLEAN JSON
# ---------------------------------------------------

def clean_json(txt):
    txt = txt.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(txt)
    except:
        if "[" in txt and "]" in txt:
            try:
                sub = txt[txt.index("[") : txt.rindex("]") + 1]
                return json.loads(sub)
            except:
                pass
        return None


# ---------------------------------------------------
# FAST LLM CALL
# ---------------------------------------------------

async def llm_call_fast(prompt: str):
    global llm_fast

    if llm_fast is None:
        llm_fast = GoogleGenAI(
            model="models/gemini-2.0-flash-lite",
            api_key=GEMINI_API_KEY
        )

    start = time.perf_counter()

    res = await llm_fast.acomplete(prompt)
    text = res.text

    duration = round(time.perf_counter() - start, 4)
    log.info(f"⚡ FAST LLM Time: {duration} sec")

    return text


# ---------------------------------------------------
# STARTUP EVENT → EMBEDDINGS
# ---------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global index

    try:
        log.info("📂 Loading documents from /pdfs ...")

        if not os.path.exists("pdfs"):
            os.makedirs("pdfs")

        docs = SimpleDirectoryReader(
            "pdfs", required_exts=[".pdf", ".txt"]
        ).load_data()

        log.info(f"🧠 Generating embeddings for {len(docs)} documents...")

        embed = GoogleGenAIEmbedding(
            model="models/text-embedding-004",
            api_key=GEMINI_API_KEY
        )

        start = time.perf_counter()

        index = VectorStoreIndex.from_documents(
            docs,
            embed_model=embed
        )

        duration = round(time.perf_counter() - start, 4)
        log.info(f"✅ Embeddings completed in {duration} seconds")
        log.info(f"📌 Vector store: {type(index._vector_store)}")
        log.info("✨ FAST RAG Engine Ready")

    except Exception as e:
        log.error("Fatal startup error:", exc_info=True)
        raise


# ---------------------------------------------------
# GET TOPICS
# ---------------------------------------------------

@app.get("/topics")
async def get_topics():
    prompt = """
    Return a JSON array of chemistry topics.
    Example: ["Organic Chemistry", "Inorganic Chemistry"]
    """

    try:
        raw = await llm_call_fast(prompt)
        data = clean_json(raw)

        if data:
            return {"topics": data}

        return {"topics": ["Fallback Topic"]}

    except Exception as e:
        log.error("Topic retrieval error:", exc_info=True)
        return {"topics": ["Fallback Topic"]}


# ---------------------------------------------------
# GENERATE QUIZ (RAG + LLM)
# ---------------------------------------------------

@app.post("/start_quiz")
async def start_quiz(req: QuizRequest):
    global index

    # 1. Retrieval
    try:
        retriever = index.as_retriever(similarity_top_k=5)

        start_ret = time.perf_counter()
        nodes = retriever.retrieve(req.topic)
        retrieval_time = round(time.perf_counter() - start_ret, 4)

        log.info(f"🔎 RAG Retrieval Time: {retrieval_time} sec")
        log.info(f"📚 Retrieved {len(nodes)} nodes")

    except Exception:
        log.error("Retrieval error:", exc_info=True)
        retrieval_time = None
        nodes = []

    # 2. LLM quiz generation
    prompt = f"""
    Generate 10 MCQ questions in JSON format.

    Topic: "{req.topic}"
    Difficulty: "{req.difficulty}"

    Use the following context if relevant:
    { [n.text for n in nodes] }

    Format:
    [
      {{"q": "...", "A": "...", "B": "...", "C": "...", "D": "...", "correct": "A"}}
    ]
    """

    try:
        raw = await llm_call_fast(prompt)
        quiz = clean_json(raw)

        if quiz and len(quiz) == 10:
            return {
                "quiz": quiz,
                "retrieval_time": retrieval_time
            }

        raise HTTPException(status_code=500, detail="Invalid quiz JSON")

    except Exception as e:
        log.error("Quiz error:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------
# FINAL SCORE
# ---------------------------------------------------

@app.post("/final_score")
async def final_score(req: ScoreRequest):
    score = sum(1 for x in req.answers if x.selected == x.correct)
    return {"score": score, "total": len(req.answers)}
