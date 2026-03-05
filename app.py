import os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.search import RAGSearch, RetrievalResult

import uvicorn

load_dotenv()

# Global variable for RAG system
rag_search: Optional[RAGSearch] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_search
    try:
        persist_dir = os.getenv("PERSIST_DIR", "faiss_store")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

        rag_search = RAGSearch(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            llm_model=llm_model,
        )
        print("[INFO] RAG system loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load RAG system: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown (cleanup if needed)
    print("[INFO] Shutting down RAG system")

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="RAG Question Answering API",
    description="FAISS + SentenceTransformers + Groq LLM",
    version="2.0.0",
    lifespan=lifespan
)

# CORS for React/Node clients
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(     
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins] if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request / Response Models
# -------------------------
class SourceItem(BaseModel):
    index: int
    distance: float
    text: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "RAG API is running. Go to /docs"}

@app.get("/health")
def health():
    if not rag_search:
        return {"ready": False}
    meta_count = len(rag_search.vectorstore.metadata) if rag_search.vectorstore else 0
    return {
        "ready": True,
        "persist_dir": rag_search.vectorstore.persist_dir,
        "documents_indexed": meta_count,
        "embedding_model": rag_search.embedding_model,
        "llm_model": rag_search.llm_model,
    }

@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    if not rag_search:
        raise HTTPException(status_code=503, detail="RAG system not ready")

    try:
        # Retrieve and summarize
        sources: List[RetrievalResult] = rag_search.retrieve(payload.query, top_k=payload.top_k)
        answer: str = rag_search.summarize(payload.query, sources)

        # Map sources for response
        resp_sources = [
            SourceItem(index=s.index, distance=float(s.distance), text=s.text)
            for s in sources
        ]
        return QueryResponse(query=payload.query, answer=answer, sources=resp_sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=True
    )