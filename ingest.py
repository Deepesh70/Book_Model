#!/usr/bin/env python3
"""
ingest.py  -  Local PDF Ingestion Pipeline

Scans theory_books/ for PDFs organized by subject:

    theory_books/
        Data_Structures/
            dsa_textbook.pdf
        Operating_Systems/
            os_concepts.pdf
        loose_file.pdf          # subject = "_general"

Extracts text with PyMuPDF (fitz), chunks with RecursiveCharacterTextSplitter,
and INCREMENTALLY appends new content into the FAISS vector store.

Usage:
    python ingest.py                               # ingest all new PDFs
    python ingest.py --subject "Data_Structures"   # one subject only
    python ingest.py --rebuild                     # wipe index, full rebuild
    python ingest.py --list                        # show all subjects & PDFs
"""

from __future__ import annotations
import argparse, pickle, sys
import numpy as np
from pathlib import Path
from typing import Optional

import fitz
import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document

from src.vectorstore import FaissVectorStore
from src.embedding import EmbeddingPipeline

load_dotenv()
BASE_DIR = Path(__file__).parent.resolve()


def _params():
    with open(BASE_DIR / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Directory scanning ────────────────────────────────────────────────────────

def _discover_pdfs(books_dir, subject_filter=None):
    """Walk theory_books/ and return [{path, name, subject}, ...]"""
    if not books_dir.exists():
        print(f"[ERROR] {books_dir} not found. Create it with subject subfolders.")
        return []
    found = []
    # Root-level PDFs
    if not subject_filter:
        for pdf in sorted(books_dir.glob("*.pdf")):
            found.append({"path": pdf, "name": pdf.name, "subject": "_general"})
    # Subject subdirectories
    for subdir in sorted(books_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subject_filter and subdir.name.lower() != subject_filter.lower():
            continue
        for pdf in sorted(subdir.rglob("*.pdf")):
            found.append({"path": pdf, "name": pdf.name, "subject": subdir.name})
    return found


def _list_subjects(books_dir):
    pdfs = _discover_pdfs(books_dir)
    if not pdfs:
        print("No PDFs found in theory_books/.")
        return
    subjects = {}
    for p in pdfs:
        subjects.setdefault(p["subject"], []).append(p["name"])
    print(f"\nFound {len(pdfs)} PDF(s) across {len(subjects)} subject(s):\n")
    for subj in sorted(subjects):
        label = subj if subj != "_general" else "(root - no subject folder)"
        print(f"  {label}/")
        for name in subjects[subj]:
            print(f"    - {name}")
    print()


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_pdf(pdf_path, subject):
    docs = []
    try:
        with fitz.open(str(pdf_path)) as pdf:
            total = len(pdf)
            for i, page in enumerate(pdf):
                text = page.get_text("text").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "subject": subject,
                            "page": i + 1,
                            "total_pages": total,
                        },
                    ))
    except Exception as exc:
        print(f"[ERROR] {pdf_path.name}: {exc}")
    return docs


# ── Registry (incremental) ───────────────────────────────────────────────────

def _load_registry(persist_dir):
    p = Path(persist_dir) / "ingested_files.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else set()

def _save_registry(persist_dir, registry):
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(persist_dir) / "ingested_files.pkl", "wb") as f:
        pickle.dump(registry, f)

def _file_key(subject, filename):
    return f"{subject}/{filename}"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_ingestion(subject=None, rebuild=False):
    cfg   = _params()
    emb_c = cfg.get("rag", {}).get("embedding", {})
    vs_c  = cfg.get("rag", {}).get("vectorstore", {})
    ing_c = cfg.get("ingestion", {})

    books_dir     = BASE_DIR / ing_c.get("books_dir", "theory_books")
    persist_dir   = vs_c.get("persist_dir", "faiss_store")
    emb_model     = emb_c.get("model", "all-MiniLM-L6-v2")
    chunk_size    = emb_c.get("chunk_size", 1000)
    chunk_overlap = emb_c.get("chunk_overlap", 200)

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  RANKLEN  --  Local PDF Ingestion Pipeline")
    print(SEP)

    # 1) Discover
    scope = f" (subject: {subject})" if subject else ""
    print(f"\n[1/4] Scanning theory_books/{scope} ...")
    pdf_list = _discover_pdfs(books_dir, subject_filter=subject)
    if not pdf_list:
        print("[WARN] No PDFs found. Add them to theory_books/<Subject>/")
        return
    print(f"  Found {len(pdf_list)} PDF(s)")

    # 2) Registry
    print("[2/4] Checking ingestion registry ...")
    if rebuild:
        ingested = set()
        for fn in ["faiss.index", "metadata.pkl", "ingested_files.pkl"]:
            fp = Path(persist_dir) / fn
            if fp.exists():
                fp.unlink()
                print(f"  removed: {fn}")
        print("  Index wiped for full rebuild.")
    else:
        ingested = _load_registry(persist_dir)
        print(f"  {len(ingested)} file(s) already indexed (will be skipped)")

    new_pdfs = []
    for item in pdf_list:
        key = _file_key(item["subject"], item["name"])
        if key in ingested:
            print(f"  skip : [{item['subject']}] {item['name']}")
        else:
            new_pdfs.append(item)

    if not new_pdfs:
        print("\n[INFO] FAISS index is already up to date.")
        return
    print(f"\n  {len(new_pdfs)} new PDF(s) to ingest")

    # 3) Extract
    print(f"[3/4] Extracting text from {len(new_pdfs)} PDF(s) via PyMuPDF ...")
    all_pages = []
    for item in new_pdfs:
        docs = _extract_pdf(item["path"], item["subject"])
        all_pages.extend(docs)
        print(f"  [{item['subject']}] {item['name']}: {len(docs)} page(s)")
    if not all_pages:
        print("[WARN] No text extracted. Aborting.")
        return

    # 4) Chunk -> Embed -> Append
    print(f"[4/4] Chunking & embedding {len(all_pages)} pages ...")
    pipe = EmbeddingPipeline(
        model_name=emb_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = pipe.chunk_documents(all_pages)
    embeds, valid = pipe.embed_chunks(chunks)
    if not valid:
        print("[WARN] No valid embeddings generated.")
        return

    vs = FaissVectorStore(
        persist_dir=persist_dir, embedding_model=emb_model,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    if (Path(persist_dir) / "faiss.index").exists() and not rebuild:
        print("[INFO] Loading existing FAISS index for incremental append ...")
        vs.load()

    metas = [{"texts": c.page_content, **c.metadata} for c in valid]
    vs.add_embeddings(np.array(embeds).astype("float32"), metas)
    vs.save()

    for item in new_pdfs:
        ingested.add(_file_key(item["subject"], item["name"]))
    _save_registry(persist_dir, ingested)

    subjs = set(item["subject"] for item in new_pdfs)
    print(f"\n{SEP}")
    print(f"  Done!")
    print(f"  Subjects processed : {', '.join(sorted(subjs))}")
    print(f"  PDFs ingested      : {len(new_pdfs)}")
    print(f"  Chunks embedded    : {len(valid)}")
    print(f"  Total in index     : {len(ingested)}")
    print(f"  FAISS store        : ./{persist_dir}/")
    print(f"{SEP}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Ingest local PDFs from theory_books/ into the FAISS vector store."
    )
    ap.add_argument("--subject", default=None,
                    help="Only ingest from this subject folder (e.g. Data_Structures)")
    ap.add_argument("--rebuild", action="store_true",
                    help="Wipe existing index and rebuild from scratch")
    ap.add_argument("--list", action="store_true", dest="list_subjects",
                    help="List all subjects and PDFs, then exit")
    args = ap.parse_args()

    if args.list_subjects:
        books_dir = BASE_DIR / _params().get("ingestion", {}).get("books_dir", "theory_books")
        _list_subjects(books_dir)
        return

    try:
        run_ingestion(subject=args.subject, rebuild=args.rebuild)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)


if __name__ == "__main__":
    main()
