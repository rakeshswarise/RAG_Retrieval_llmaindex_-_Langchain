import os
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================================================
# LOGGING (FILE + CONSOLE)
# =====================================================

if not os.path.exists("logs"):
    os.makedirs("logs")

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler = RotatingFileHandler(
    "logs/rag_fast.log",
    maxBytes=5_000_000,
    backupCount=3,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

log = logging.getLogger("rag-fast")
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log.addHandler(console_handler)

# =====================================================
# FASTAPI SETUP
# =====================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# KEYS + ENV SETUP
# =====================================================

GEMINI_API_KEY = ""
PINECONE_API_KEY = ""

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

PINECONE_INDEX_NAME = "quizz"
PINECONE_REGION = "us-east-1"

# =====================================================
# IMPORTS
# =====================================================

from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Global holders
vector_store = None
llm_fast = None

# =====================================================
# MODELS
# =====================================================

class QuizRequest(BaseModel):
    topic: str
    difficulty: str

class Answer(BaseModel):
    selected: str
    correct: str

class ScoreRequest(BaseModel):
    answers: List[Answer]


# =====================================================
# JSON CLEANER
# =====================================================

def clean_json(txt):
    txt = txt.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(txt)
    except:
        if "[" in txt and "]" in txt:
            sub = txt[txt.index("["): txt.rindex("]") + 1]
            try:
                return json.loads(sub)
            except:
                pass
        return None


# =====================================================
# FAST LLM
# =====================================================

async def llm_call_fast(prompt: str):
    global llm_fast

    if llm_fast is None:
        llm_fast = GoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=GEMINI_API_KEY
        )

    start = time.perf_counter()
    res = await llm_fast.ainvoke(prompt)
    duration = round(time.perf_counter() - start, 4)
    log.info(f"⚡ FAST LLM Time: {duration} sec")
    return res


# =====================================================
# STARTUP → BUILD PINECONE VECTOR STORE
# =====================================================

@app.on_event("startup")
async def startup_event():
    global vector_store

    try:
        log.info("📂 Loading documents from /pdfs ...")

        if not os.path.exists("pdfs"):
            os.makedirs("pdfs")

        docs = []
        for file in os.listdir("pdfs"):
            path = os.path.join("pdfs", file)
            if file.endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
            elif file.endswith(".txt"):
                docs.extend(TextLoader(path).load())

        log.info(f"📄 Loaded {len(docs)} files")

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100
        )
        split_docs = splitter.split_documents(docs)
        log.info(f"🧩 Split into {len(split_docs)} chunks")

        # Embeddings
        embed = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GEMINI_API_KEY
        )

        # Pinecone client
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(PINECONE_INDEX_NAME)
        log.info("🔌 Connected to Pinecone index")

        start_vec = time.perf_counter()

        # Create vector store
        vector_store = PineconeVectorStore.from_documents(
            split_docs,
            embedding=embed,
            index_name=PINECONE_INDEX_NAME
        )

        duration = round(time.perf_counter() - start_vec, 4)
        log.info(f"🔐 Pinecone ready in {duration} sec")
        log.info("✨ FAST RAG Engine Ready")

    except Exception:
        log.error("Fatal startup error:", exc_info=True)
        raise


# =====================================================
# GET TOPICS
# =====================================================

@app.get("/topics")
async def get_topics():
    prompt = """
    Return a JSON array of chemistry topics.
    Example: ["Organic Chemistry", "Inorganic Chemistry"]
    """
    try:
        raw = await llm_call_fast(prompt)
        data = clean_json(raw)
        return {"topics": data or ["Fallback Topic"]}
    except:
        log.error("Topic error:", exc_info=True)
        return {"topics": ["Fallback Topic"]}


# =====================================================
# START QUIZ
# =====================================================

@app.post("/start_quiz")
async def start_quiz(req: QuizRequest):
    global vector_store

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    start_ret = time.perf_counter()
    nodes = retriever.invoke(req.topic)
    retrieval_time = round(time.perf_counter() - start_ret, 4)

    log.info(f"🔎 Retrieval Time: {retrieval_time} sec")
    log.info(f"📚 Retrieved {len(nodes)} chunks")

    context = "\n\n".join([n.page_content for n in nodes])

    prompt = f"""
    Generate 10 MCQ questions in JSON format.

    Topic: "{req.topic}"
    Difficulty: "{req.difficulty}"

    Context:
    {context}

    Format:
    [
        {{"q": "...", "A": "...", "B": "...", "C": "...", "D": "...", "correct": "A"}}
    ]
    """
    try:
        raw = await llm_call_fast(prompt)
        quiz = clean_json(raw)
        if quiz and len(quiz) == 10:
            return {"quiz": quiz, "retrieval_time": retrieval_time}
        raise HTTPException(status_code=500, detail="Invalid JSON")

    except Exception as e:
        log.error("Quiz generation error:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# SCORE
# =====================================================

@app.post("/final_score")
async def final_score(req: ScoreRequest):
    score = sum(1 for x in req.answers if x.selected == x.correct)
    return {"score": score, "total": len(req.answers)}