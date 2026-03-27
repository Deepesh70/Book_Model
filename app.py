import os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.search import RAGSearch, RetrievalResult
from generator import generate_theory_page

import uvicorn

load_dotenv()

# Global variable for existing RAG system
rag_search: Optional[RAGSearch] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_search
    try:
        persist_dir     = os.getenv("PERSIST_DIR", "faiss_store")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        llm_model       = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        rag_search = RAGSearch(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            llm_model=llm_model,
        )
        print("[INFO] RAG system loaded successfully")
    except Exception as e:
        print(f"[WARN] RAG system failed to load: {e}. Server still starts.")
        rag_search = None
    yield
    print("[INFO] Shutting down")

# FastAPI app
app = FastAPI(
    title="Ranklen RAG API",
    description="FAISS + SentenceTransformers + Groq/Gemini  |  /query  /generate-theory",
    version="3.0.0",
    lifespan=lifespan,
    root_path=os.getenv("ROOT_PATH", ""),
)

cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins] if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────
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

class TheoryRequest(BaseModel):
    topic: str
    provider: Optional[str] = None  # None -> uses params.yaml default ("gemini")

class TheoryResponse(BaseModel):
    success: bool
    topic: str
    content_markdown: str
    provider_used: str

# ── Existing routes (unchanged) ───────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Ranklen RAG API is running. Go to /docs"}

@app.get("/health")
def health():
    if not rag_search:
        return {"ready": False, "note": "RAG system not loaded (run ingest.py first)"}
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
        raise HTTPException(
            status_code=503,
            detail="RAG system not ready. Run ingest.py first.",
        )
    try:
        sources: List[RetrievalResult] = rag_search.retrieve(
            payload.query, top_k=payload.top_k
        )
        answer: str = rag_search.summarize(payload.query, sources)
        resp_sources = [
            SourceItem(index=s.index, distance=float(s.distance), text=s.text)
            for s in sources
        ]
        return QueryResponse(query=payload.query, answer=answer, sources=resp_sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── NEW: Theory Synthesis endpoint ────────────────────────────────────────────
@app.post("/generate-theory", response_model=TheoryResponse)
def generate_theory(payload: TheoryRequest):
    """
    Synthesize an original educational article about the given topic.

    Flow:
      1. Queries FAISS for relevant book excerpts (top-k=8 by default).
      2. Checks L2 distance against threshold (params.yaml: distance_threshold).
         - Below threshold  -> RAG synthesis (book context provided to LLM).
         - Above threshold  -> Fallback (LLM uses its general knowledge only).
      3. FAISS index missing -> graceful fallback (server does NOT crash).

    Request body:
        { "topic": "Binary Search Trees" }
        { "topic": "Sorting Algorithms", "provider": "groq" }  // optional override

    Response:
        { "success": true, "topic": "...", "content_markdown": "...", "provider_used": "..." }
    """
    if not payload.topic or not payload.topic.strip():
        raise HTTPException(status_code=422, detail="topic must be a non-empty string")

    topic    = payload.topic.strip()
    provider = (payload.provider or "").strip() or None  # None -> params.yaml default

    # Resolve which provider will be used (for the response field)
    import yaml
    from pathlib import Path as _Path
    try:
        cfg_provider = (
            yaml.safe_load(
                open(_Path(__file__).parent / "params.yaml", encoding="utf-8")
            )
            .get("theory_synthesis", {})
            .get("default_provider", "gemini")
        )
    except Exception:
        cfg_provider = "gemini"
    resolved_provider = provider or cfg_provider

    try:
        markdown = generate_theory_page(topic=topic, provider=provider)
        return TheoryResponse(
            success=True,
            topic=topic,
            content_markdown=markdown,
            provider_used=resolved_provider,
        )
    except ValueError as ve:
        # Missing API key
        raise HTTPException(status_code=503, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(exc)}")


# ── Local run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=True,
    )
