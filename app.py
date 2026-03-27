import os
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.search import RAGSearch, RetrievalResult
from generator import generate_theory_page, SynthesisResult

import uvicorn

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
)
logger = logging.getLogger("ranklen.api")

# Global RAG system (for /query endpoint)
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
        logger.info("RAG system loaded successfully")
    except Exception as e:
        logger.warning(f"RAG system failed to load: {e}. Server still starts.")
        rag_search = None
    yield
    logger.info("Shutting down")


# FastAPI app
app = FastAPI(
    title="Ranklen RAG API v4",
    description=(
        "Enterprise-grade RAG pipeline with LLM Query Classification + "
        "Subject-Isolated FAISS Search + Theory Synthesis.\n\n"
        "Endpoints: /query (legacy) | /generate-theory (classified + filtered)"
    ),
    version="4.0.0",
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
    provider: Optional[str] = None

class TheoryResponse(BaseModel):
    success: bool
    topic: str
    content_markdown: str
    # ── Classification metadata (new in v4) ─────────
    classified_subject: str                # "DBMS", "Os", "Unknown"
    search_mode: str                        # "rag_filtered", "fallback_unknown", etc.
    provider_used: str
    chunks_retrieved: int
    min_distance: float
    available_subjects: List[str]
    classification_time_ms: float
    synthesis_time_ms: float


# ── Existing routes ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Ranklen RAG API v4 is running. Go to /docs"}

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
        raise HTTPException(status_code=503, detail="RAG system not ready.")
    try:
        sources: List[RetrievalResult] = rag_search.retrieve(payload.query, top_k=payload.top_k)
        answer: str = rag_search.summarize(payload.query, sources)
        resp_sources = [
            SourceItem(index=s.index, distance=float(s.distance), text=s.text)
            for s in sources
        ]
        return QueryResponse(query=payload.query, answer=answer, sources=resp_sources)
    except Exception as exc:
        logger.error(f"/query error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Theory Synthesis (Classify → Filter → Synthesize) ─────────────────────────

@app.post("/generate-theory", response_model=TheoryResponse)
def generate_theory(payload: TheoryRequest):
    """
    Enterprise-grade theory synthesis endpoint.

    Pipeline:
      1. CLASSIFY — LLM classifies the query into an available subject.
      2. FILTER   — Subject-isolated FAISS search (only matching chunks).
      3. GATE     — Deterministic L2 distance threshold check.
      4. SYNTH    — RAG synthesis (with context) or fallback (general knowledge).

    Request:
        { "topic": "Explain B+ tree indexing" }
        { "topic": "Process scheduling", "provider": "groq" }

    Response includes full classification metadata for observability.
    """
    if not payload.topic or not payload.topic.strip():
        raise HTTPException(status_code=422, detail="topic must be a non-empty string")

    topic    = payload.topic.strip()
    provider = (payload.provider or "").strip() or None

    try:
        result: SynthesisResult = generate_theory_page(
            topic=topic, provider=provider,
        )
        return TheoryResponse(
            success=True,
            topic=topic,
            content_markdown=result.markdown,
            classified_subject=result.classified_subject,
            search_mode=result.search_mode,
            provider_used=result.provider_used,
            chunks_retrieved=result.chunks_retrieved,
            min_distance=result.min_distance,
            available_subjects=result.available_subjects,
            classification_time_ms=result.classification_time_ms,
            synthesis_time_ms=result.synthesis_time_ms,
        )
    except ValueError as ve:
        logger.error(f"Config/API key error: {ve}")
        raise HTTPException(status_code=503, detail=str(ve))
    except Exception as exc:
        logger.error(f"Synthesis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(exc)}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=True,
    )
