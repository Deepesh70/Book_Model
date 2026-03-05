---
title: Book Model - RAG QA API
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Book Model — RAG Question Answering API

A Retrieval-Augmented Generation (RAG) API built with **FastAPI**, **FAISS**, **SentenceTransformers**, and **Groq LLM**.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health message |
| `GET` | `/health` | Detailed system status |
| `POST` | `/query` | Ask a question against the indexed documents |
| `GET` | `/docs` | Interactive Swagger UI |

### Example — Query the API

```bash
curl -X POST "https://jeevant010-book-model.hf.space/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 3}'
```

### Response

```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is ...",
  "sources": [
    {"index": 0, "distance": 0.42, "text": "..."}
  ]
}
```

## Environment Variables

Set `GROQ_API_KEY` as a **Secret** in your Hugging Face Space settings.

## Tech Stack

- **FastAPI** — async web framework
- **FAISS** — vector similarity search
- **SentenceTransformers** — embedding model (`all-MiniLM-L6-v2`)
- **Groq** — LLM inference (`llama-3.1-8b-instant`)
- **LangChain** — document loading & text splitting