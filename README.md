# RAG Retrieval LLAMA INDEX AND LangChain (LOCAL & PINCONE VECTOR DB)

- langchain_faiss/

- langchain_pinecone/

- LLamaindex_local/

- LLamaindex_pinecone/



# main.py = backend server

# index.html = frontend UI

# logs/ = logging

# pdfs/ = document input


# RAG Quiz Engine – Multi-Backend Comparison (FAISS, Pinecone, LangChain, LlamaIndex)

This repository contains four implementations of a RAG-based Quiz Generation Engine using different vector database and framework combinations. Each folder represents a complete, independent RAG pipeline with its own backend server (main.py) and frontend interface (index.html).

- The goal of this project is to compare performance between:

- LangChain + FAISS (local)

- LangChain + Pinecone (cloud)

- LlamaIndex + Local Vector Store

- LlamaIndex + Pinecone (cloud)

All four implementations expose the same REST API, use the same UI, and process the same input documents. This allows accurate benchmarking of retrieval performance, LLM latency, and indexing time.

1. Project Structure
```bash
QUIZ_ALL/
│
├── langchain_faiss/
│   ├── main.py
│   ├── index.html
│   ├── pdfs/
│   ├── logs/
│   └── langchain_faiss.txt     (performance results)
│
├── langchain_pinecone/
│   ├── main.py
│   ├── index.html
│   ├── pdfs/
│   ├── logs/
│   └── langchain_pinecone.txt  (performance results)
│
├── LLamaindex_local/
│   ├── main.py
│   ├── index.html
│   ├── pdfs/
│   ├── logs/
│   └── llamaindex_local.txt    (performance results)
│
├── LLamaindex_pinecone/
│   ├── main.py
│   ├── index.html
│   ├── pdfs/
│   ├── logs/
│   └── llamaindex_pinecone.txt (performance results)
│
└── README.md  (this document)
```

# 2. How Each Engine Works

All four engines follow the same core pipeline:

Load PDFs/TXT files from the pdfs/ directory

Split documents into chunks

Generate embeddings using Gemini text-embedding-004

Store embeddings in:

- FAISS (local)

- Local vector store (LlamaIndex)

- Pinecone (cloud)

- Query the vector store

- Generate MCQs using Gemini 2.0 Flash Lite

The frontend sends HTTP requests to the backend, receives MCQs, and displays them.

# 3. Running Any Engine

``` Step 1: Install dependencies ```

- Inside each engine folder:

- pip install -r requirements.txt


- (Ensure FastAPI, Uvicorn, LangChain, LlamaIndex, Pinecone, and Google Generative AI packages are installed.)

``` Step 2: Add your PDFs ```

Place all PDF or TXT files inside:

/pdfs

 ``` Step 3: Start the API server ```

``` bash uvicorn main:app --reload --port 8000  ```

``` Step 4: Open the UI ```

Open the following file in a browser:

open ```  index.html ```


The UI connects to the FastAPI backend and displays topics, quiz questions, and final score.

# 4. API Endpoints
GET /topics

Returns a list of chemistry topics.

POST /start_quiz

Input:

{
  "topic": "Organic Chemistry",
  "difficulty": "medium"
}


Output:

# 10 MCQs in JSON format

Retrieval time (seconds)

POST /final_score

Input:

{
  "answers": [
    {"selected": "A", "correct": "A"},
    {"selected": "B", "correct": "C"}
  ]
}


Output:

{"score": 7, "total": 10}

# 5. Performance Comparison

Benchmark results were collected using the same dataset (164 documents, 314 chunks).

Performance Summary
| RAG Setup                           | Index Build Time | Retrieval Time | LLM Time (Avg) | Notes                                      |
| ----------------------------------- | ---------------- | -------------- | -------------- | ------------------------------------------ |
| **LangChain + FAISS (Local)**       | **9.09 sec**     | **0.425 sec**  | ~3 sec         | Fastest overall; ideal for local workloads |
| **LlamaIndex + Local Vector Store** | **11.20 sec**    | **0.523 sec**  | ~3 sec         | Stable & fast; good abstraction            |
| **LlamaIndex + Pinecone**           | **86.05 sec**    | **1.304 sec**  | ~3 sec         | Good scalability; slow indexing            |
| **LangChain + Pinecone**            | **19.80 sec**    | **3.463 sec**  | ~3 sec         | Highest retrieval latency                  |


Local vector stores (FAISS, LlamaIndex local) are significantly faster than Pinecone for small–medium document sets.

Pinecone introduces network latency, making retrieval slower.

LLM time is identical across all setups (Gemini API).

# 6. Recommendations
Best for Speed

LangChain + FAISS

Best for LlamaIndex users

LlamaIndex + Local Vector Store

Use Pinecone when:

Your dataset exceeds 1 million vectors

You require cloud hosting or shared access

You need horizontal scalability

# 7. Logging

Each implementation writes logs to:

/logs/rag_fast.log


Using a rotating file handler with:

5 MB max size

3 backup files

Logs include:

Indexing time

Retrieval time

LLM duration

Error stack traces

# 8. Environment Variables

Each backend requires:

- GEMINI_API_KEY
- PINECONE_API_KEY  (only required for Pinecone versions)


These can be stored in:

.env

OS environment variables

Directly in the script (not recommended)

# 9. Frontend (index.html)

All implementations share the same UI:

- Fetches topics

- Starts quiz

- Displays MCQs

- Submits answers

- Shows final score

Communicates with backend using REST APIs

This ensures fair performance comparison.