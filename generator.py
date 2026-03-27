"""
generator.py  -  Ranklen Theory Synthesis Engine v2.0

Architecture:
    1. classify_query_intent()  →  LLM classifies query into a subject
    2. filtered_query()         →  FAISS search filtered by that subject
    3. threshold check          →  deterministic distance gate
    4. synthesize               →  RAG or fallback LLM call

This guarantees subject-isolated vector searches:
  "Explain B+ trees" → classified as "DBMS" → only DBMS chunks searched.
  "Page replacement"  → classified as "Os"   → only OS chunks searched.
  "Quantum physics"   → classified as "Unknown" → skip FAISS, use LLM knowledge.

LLM provider is flexible (gemini/groq), configurable in params.yaml.
"""

from __future__ import annotations
import os, time, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.vectorstore import FaissVectorStore

load_dotenv()
BASE_DIR = Path(__file__).parent.resolve()
logger = logging.getLogger("ranklen.generator")


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

def _params() -> dict:
    with open(BASE_DIR / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Result dataclass — returned by generate_theory_page
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SynthesisResult:
    """Structured result from the synthesis pipeline."""
    markdown: str
    topic: str
    classified_subject: str              # "DBMS", "Os", "Unknown"
    search_mode: str                      # "rag_filtered", "fallback_unknown", "fallback_threshold", "fallback_no_index"
    provider_used: str
    chunks_retrieved: int = 0
    min_distance: float = -1.0
    available_subjects: List[str] = field(default_factory=list)
    classification_time_ms: float = 0.0
    synthesis_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Factory (flexible provider)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_llm(provider: str, cfg: dict, temperature_override: Optional[float] = None):
    """
    Return a LangChain chat-model.
    Supported: 'gemini', 'groq'. Extend here for more.
    """
    provider = provider.lower().strip()
    temp = temperature_override if temperature_override is not None else cfg.get("temperature", 0.4)

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set. Add it to .env.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=cfg.get("gemini_model", "gemini-2.0-flash"),
            google_api_key=api_key,
            temperature=temp,
        )
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Add it to .env.")
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=cfg.get("groq_model", "llama-3.3-70b-versatile"),
            api_key=api_key,
            temperature=temp,
        )
    else:
        raise ValueError(f"Unknown provider '{provider}'. Supported: 'gemini', 'groq'.")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: LLM Query Classifier
# ═══════════════════════════════════════════════════════════════════════════════

CLASSIFIER_SYSTEM = (
    "You are a strict query classifier for an educational system. "
    "Given a user's query and a list of available curriculum subjects, "
    "determine which single subject the query belongs to.\n\n"
    "RULES:\n"
    "1. Respond with ONLY the exact subject name from the list. No explanation.\n"
    "2. If the query does not clearly match any subject, respond with exactly: Unknown\n"
    "3. Be generous in matching — 'database normalization' matches 'DBMS', "
    "'process scheduling' matches 'Os', etc.\n"
    "4. Output a single word or phrase. Nothing else."
)

CLASSIFIER_HUMAN = (
    "Available subjects: {subjects}\n\n"
    "User query: {query}\n\n"
    "Classification:"
)


def classify_query_intent(
    user_query: str,
    available_subjects: List[str],
    provider: str = "gemini",
    cfg: dict = None,
) -> str:
    """
    Classify a user query into one of the available subjects using a fast LLM call.

    Parameters
    ----------
    user_query         : The user's educational query.
    available_subjects : List of subject names from the FAISS index (e.g. ["DBMS", "Os"]).
    provider           : LLM provider for classification.
    cfg                : Config dict (theory_synthesis section from params.yaml).

    Returns
    -------
    str  Exact subject name from available_subjects, or "Unknown".
    """
    if not available_subjects:
        logger.warning("No subjects available for classification. Returning 'Unknown'.")
        return "Unknown"

    cfg = cfg or {}
    classifier_cfg = cfg.copy()

    # Use very low temperature for deterministic classification
    llm = _get_llm(provider, classifier_cfg, temperature_override=0.05)

    subjects_str = ", ".join(sorted(available_subjects))
    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM),
        HumanMessage(content=CLASSIFIER_HUMAN.format(
            subjects=subjects_str, query=user_query
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw_output = response.content.strip()

        # Strict validation: must be an exact match (case-insensitive)
        for subj in available_subjects:
            if raw_output.lower() == subj.lower():
                logger.info(f"Query classified: '{user_query}' -> {subj}")
                return subj

        # Partial match fallback (e.g. LLM returns "DBMS (Database)" but
        # we have "DBMS" — check if any subject is a substring)
        for subj in available_subjects:
            if subj.lower() in raw_output.lower():
                logger.info(f"Query classified (partial match): '{user_query}' -> {subj}")
                return subj

        logger.info(f"Query unclassified: '{user_query}' -> Unknown (LLM said: '{raw_output}')")
        return "Unknown"

    except Exception as exc:
        logger.error(f"Classification failed: {exc}. Defaulting to 'Unknown'.")
        return "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Synthesis Prompts
# ═══════════════════════════════════════════════════════════════════════════════

RAG_SYSTEM = (
    "You are an elite technical writer and educator. "
    "You will receive textbook excerpts from multiple academic sources on a specific subject. "
    "Synthesize them into a single, comprehensive, entirely ORIGINAL educational article. "
    "CRITICAL RULES: "
    "(1) Do NOT copy-paste any sentence verbatim. "
    "(2) Rewrite every concept entirely in your own words. "
    "(3) Structure with clear Markdown headings, subheadings, and examples. "
    "(4) Output clean, well-formatted Markdown only."
)

RAG_HUMAN = (
    "Subject: {subject}\n"
    "Topic: {topic}\n\n"
    "Textbook Excerpts ({n_chunks} chunks from {subject} books):\n"
    "{context}\n\n"
    "Synthesize an original, in-depth educational article about the topic above. "
    "Use the excerpts as reference material but rewrite everything in your own words."
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


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: Core Synthesis — Classify → Filter → Synthesize
# ═══════════════════════════════════════════════════════════════════════════════

def generate_theory_page(
    topic: str,
    provider: Optional[str] = None,
) -> SynthesisResult:
    """
    Generate an original educational Markdown article.

    Pipeline:
      1. Load FAISS index and discover available subjects.
      2. Classify the query into a subject via LLM.
      3. If Unknown → skip FAISS, use fallback LLM knowledge.
      4. If known subject → filtered FAISS search (subject-isolated).
      5. Threshold check on filtered results.
      6. RAG synthesis or fallback.

    Parameters
    ----------
    topic    : The educational topic (e.g. "B+ Tree indexing").
    provider : LLM provider override. None → uses params.yaml default.

    Returns
    -------
    SynthesisResult  Structured result with markdown + metadata.
    """
    cfg     = _params()
    ts_cfg  = cfg.get("theory_synthesis", {})
    vs_cfg  = cfg.get("rag", {}).get("vectorstore", {})
    emb_cfg = cfg.get("rag", {}).get("embedding", {})

    persist_dir = vs_cfg.get("persist_dir", "faiss_store")
    emb_model   = emb_cfg.get("model", "all-MiniLM-L6-v2")
    top_k       = ts_cfg.get("top_k", 8)
    threshold   = ts_cfg.get("distance_threshold", 1.0)
    provider    = provider or ts_cfg.get("default_provider", "gemini")

    result = SynthesisResult(
        markdown="", topic=topic, classified_subject="Unknown",
        search_mode="fallback_no_index", provider_used=provider,
    )

    # ── Step 1: Load FAISS index ──────────────────────────────────────────
    faiss_path = Path(persist_dir) / "faiss.index"
    vs = None
    available_subjects = []

    if faiss_path.exists():
        try:
            vs = FaissVectorStore(persist_dir=persist_dir, embedding_model=emb_model)
            vs.load()
            available_subjects = sorted(vs.get_indexed_subjects())
            result.available_subjects = available_subjects
            stats = vs.get_index_stats()
            print(f"[INFO] FAISS loaded: {stats['total_vectors']} vectors, "
                  f"subjects: {stats['subjects']}")
        except Exception as exc:
            print(f"[WARN] FAISS load failed: {exc}")
            vs = None
    else:
        print(f"[INFO] No FAISS index at ./{persist_dir}/ (run ingest.py first)")

    # ── Step 2: Classify query intent ─────────────────────────────────────
    if vs and available_subjects:
        t0 = time.time()
        classified = classify_query_intent(
            user_query=topic,
            available_subjects=available_subjects,
            provider=provider,
            cfg=ts_cfg,
        )
        result.classification_time_ms = round((time.time() - t0) * 1000, 1)
        result.classified_subject = classified
        print(f"[INFO] Classification: '{topic}' -> '{classified}' "
              f"({result.classification_time_ms}ms)")
    else:
        result.classified_subject = "Unknown"
        print("[INFO] Skipping classification (no FAISS index or no subjects)")

    # ── Step 3: Filtered FAISS search or fallback ─────────────────────────
    use_rag = False
    context = ""

    if result.classified_subject != "Unknown" and vs:
        # Subject-isolated FAISS search
        filtered_results = vs.filtered_query(
            query_text=topic,
            top_k=top_k,
            filter={"subject": result.classified_subject},
        )

        if filtered_results:
            min_dist = min(r["distance"] for r in filtered_results)
            result.min_distance = round(min_dist, 4)
            result.chunks_retrieved = len(filtered_results)

            print(f"[INFO] Filtered search: {len(filtered_results)} chunks, "
                  f"min_dist={min_dist:.4f} (threshold={threshold})")

            # Deterministic threshold check
            if min_dist < threshold:
                use_rag = True
                context = "\n\n---\n\n".join(
                    r["metadata"]["texts"]
                    for r in filtered_results
                    if r.get("metadata") and r["metadata"].get("texts")
                )
                result.search_mode = "rag_filtered"
                print(f"[INFO] RAG path active: {len(filtered_results)} "
                      f"'{result.classified_subject}' chunks in context")
            else:
                result.search_mode = "fallback_threshold"
                print(f"[INFO] Fallback: distance {min_dist:.4f} >= threshold {threshold}")
        else:
            result.search_mode = "fallback_threshold"
            print(f"[INFO] No filtered results for subject '{result.classified_subject}'")
    else:
        result.search_mode = (
            "fallback_unknown" if vs else "fallback_no_index"
        )
        print(f"[INFO] Fallback mode: {result.search_mode}")

    # ── Step 4: LLM synthesis ─────────────────────────────────────────────
    llm = _get_llm(provider, ts_cfg)

    if use_rag:
        messages = [
            SystemMessage(content=RAG_SYSTEM),
            HumanMessage(content=RAG_HUMAN.format(
                subject=result.classified_subject,
                topic=topic,
                n_chunks=result.chunks_retrieved,
                context=context,
            )),
        ]
    else:
        messages = [
            SystemMessage(content=FALLBACK_SYSTEM),
            HumanMessage(content=FALLBACK_HUMAN.format(topic=topic)),
        ]

    print(f"[INFO] Calling {provider} LLM for synthesis...")
    t0 = time.time()
    response = llm.invoke(messages)
    result.synthesis_time_ms = round((time.time() - t0) * 1000, 1)
    result.markdown = response.content

    print(f"[INFO] Synthesis complete ({result.synthesis_time_ms}ms) | "
          f"mode={result.search_mode} | subject={result.classified_subject}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) or "Normalization in DBMS"
    print(f"\nGenerating article for: {topic}\n")
    r = generate_theory_page(topic)
    print(f"\n{'='*60}")
    print(f"Subject: {r.classified_subject} | Mode: {r.search_mode}")
    print(f"Chunks: {r.chunks_retrieved} | Min Dist: {r.min_distance}")
    print(f"Classification: {r.classification_time_ms}ms | Synthesis: {r.synthesis_time_ms}ms")
    print(f"{'='*60}\n")
    print(r.markdown)
