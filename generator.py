"""
generator.py  -  Ranklen Theory Synthesis Engine

Core function: generate_theory_page(topic, provider)

Retrieves relevant chunks from the FAISS vector store, applies a
deterministic L2 distance threshold check, then calls an LLM to
synthesize an original educational article in Markdown.

Threshold logic (deterministic - no prompt engineering guesswork):
    distance < threshold  ->  RAG path  (book context provided to LLM)
    distance >= threshold ->  Fallback  (LLM uses general knowledge only)

LLM provider is fully flexible:
    default : 'gemini'  (GOOGLE_API_KEY)  <- set in params.yaml
    fallback: 'groq'    (GROQ_API_KEY)
    Switch provider by changing theory_synthesis.default_provider in params.yaml.
    No code changes needed.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.vectorstore import FaissVectorStore

load_dotenv()
BASE_DIR = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _params() -> dict:
    with open(BASE_DIR / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Flexible LLM factory
# ---------------------------------------------------------------------------

def _get_llm(provider: str, ts_cfg: dict):
    """
    Return a LangChain chat-model instance.
    Supported: 'gemini', 'groq'
    To add a new provider (e.g. 'openai'), extend this function only.
    """
    provider = provider.lower().strip()
    temp     = ts_cfg.get("temperature", 0.4)

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. "
                "Add it to Book_Model/.env or your environment variables."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=ts_cfg.get("gemini_model", "gemini-2.0-flash"),
            google_api_key=api_key,
            temperature=temp,
        )

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. "
                "Add it to Book_Model/.env or your environment variables."
            )
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=ts_cfg.get("groq_model", "llama-3.3-70b-versatile"),
            api_key=api_key,
            temperature=temp,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Supported: 'gemini', 'groq'. "
            "Add more providers in generator.py -> _get_llm()."
        )


# ---------------------------------------------------------------------------
# Synthesis prompts
# ---------------------------------------------------------------------------

RAG_SYSTEM = (
    "You are an elite technical writer and educator. "
    "You will receive textbook excerpts from multiple academic sources. "
    "Synthesize them into a single, comprehensive, entirely ORIGINAL educational article. "
    "CRITICAL RULES: "
    "(1) Do NOT copy-paste any sentence verbatim. "
    "(2) Rewrite every concept entirely in your own words. "
    "(3) Structure the article with clear headings, subheadings, and examples. "
    "(4) Output clean, well-formatted Markdown only."
)

RAG_HUMAN = (
    "Topic: {topic}\n\n"
    "Textbook Excerpts (multiple sources):\n"
    "{context}\n\n"
    "Write a comprehensive original educational article about the topic above. "
    "Use the excerpts as reference but rewrite everything in your own words."
)

FALLBACK_SYSTEM = (
    "You are an elite technical writer and educator with deep expertise across "
    "computer science, mathematics, and engineering. "
    "Write accurate, comprehensive, and well-structured educational content. "
    "Output clean, well-formatted Markdown only."
)

FALLBACK_HUMAN = (
    "Write a comprehensive, in-depth educational article about: {topic}\n\n"
    "Include: clear definitions, key concepts, real-world examples, "
    "common use cases, and important formulas or algorithms where applicable. "
    "Structure with proper Markdown headings and subheadings."
)


# ---------------------------------------------------------------------------
# Core synthesis function
# ---------------------------------------------------------------------------

def generate_theory_page(
    topic: str,
    provider: Optional[str] = None,
) -> str:
    """
    Generate an original educational Markdown article about a topic.

    Parameters
    ----------
    topic    : Subject to explain (e.g. 'Binary Search Trees').
    provider : LLM provider override ('gemini' or 'groq').
               If None, reads theory_synthesis.default_provider from params.yaml.

    Returns
    -------
    str  Markdown-formatted educational article.
    """
    cfg     = _params()
    ts_cfg  = cfg.get("theory_synthesis", {})
    vs_cfg  = cfg.get("rag", {}).get("vectorstore", {})
    emb_cfg = cfg.get("rag", {}).get("embedding", {})

    persist_dir   = vs_cfg.get("persist_dir", "faiss_store")
    emb_model     = emb_cfg.get("model", "all-MiniLM-L6-v2")
    top_k         = ts_cfg.get("top_k", 8)
    threshold     = ts_cfg.get("distance_threshold", 1.0)
    provider      = provider or ts_cfg.get("default_provider", "gemini")

    use_rag = False
    context = ""

    # ---- Attempt FAISS retrieval ----------------------------------------
    faiss_path = Path(persist_dir) / "faiss.index"
    if faiss_path.exists():
        try:
            vs = FaissVectorStore(persist_dir=persist_dir, embedding_model=emb_model)
            vs.load()
            results = vs.query(query_text=topic, top_k=top_k)

            if results:
                min_dist = min(r["distance"] for r in results)
                print(f"[INFO] FAISS min L2 distance for '{topic}': {min_dist:.4f}  (threshold={threshold})")

                # Deterministic threshold check -- no LLM prompt engineering
                if min_dist < threshold:
                    use_rag = True
                    context = "\n\n---\n\n".join(
                        r["metadata"]["texts"]
                        for r in results
                        if r.get("metadata") and r["metadata"].get("texts")
                    )
                    print(f"[INFO] RAG path active: {len(results)} chunk(s) in context")
                else:
                    print(f"[INFO] Fallback path: distance {min_dist:.4f} >= threshold {threshold}")
            else:
                print("[INFO] FAISS returned no results. Using fallback.")
        except Exception as exc:
            print(f"[WARN] FAISS error: {exc}. Using fallback.")
    else:
        print(f"[INFO] No FAISS index at ./{persist_dir}/ -- using fallback (run ingest.py first)")

    # ---- Build messages and call LLM ------------------------------------
    llm = _get_llm(provider, ts_cfg)

    if use_rag:
        messages = [
            SystemMessage(content=RAG_SYSTEM),
            HumanMessage(content=RAG_HUMAN.format(topic=topic, context=context)),
        ]
    else:
        messages = [
            SystemMessage(content=FALLBACK_SYSTEM),
            HumanMessage(content=FALLBACK_HUMAN.format(topic=topic)),
        ]

    print(f"[INFO] Calling {provider} LLM for: '{topic}'")
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) or "Binary Search Trees"
    print(f"Generating article for: {topic}")
    print(generate_theory_page(topic))
