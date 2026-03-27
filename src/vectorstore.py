import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Dict, Optional, Set
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    """
    Custom FAISS vector store with metadata payload and subject-level filtering.

    The index uses IndexFlatL2 for exact brute-force L2 search.
    Metadata (including 'subject', 'source', 'page', 'texts') is stored
    in a parallel list and persisted alongside the FAISS index.

    Filtering is implemented as over-fetch + post-filter: we retrieve
    k * multiplier results from FAISS, then narrow down by metadata match.
    This guarantees subject-isolated results without maintaining
    separate per-subject indexes.
    """

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw document(s)...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings, valid_chunks = emb_pipe.embed_chunks(chunks)

        if len(valid_chunks) == 0:
            print("[WARNING] No valid chunks were embedded. Vector store will not be updated.")
            return

        metadatas = [{"texts": chunk.page_content} for chunk in valid_chunks]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()
        print(f"[INFO] Vector Store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        if embeddings.size == 0:
            print("[WARNING] No embeddings to add. Vector store remains empty.")
            return

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss Index.")

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self):
        if self.index is None:
            print("[WARNING] Cannot save: index is empty. Skipping save operation.")
            return
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"Missing index/metadata in {self.persist_dir}. Build the store first."
            )
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss Index ({self.index.ntotal} vectors) from {self.persist_dir}")

    # ── Unfiltered search (original — unchanged) ─────────────────────────────

    def search(self, query_embeddings: np.ndarray, top_k: int = 5):
        if self.index is None:
            print("[WARNING] Vector store is empty. No results to return.")
            return []

        D, I = self.index.search(query_embeddings, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0:
                continue  # FAISS returns -1 when fewer results than top_k
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        if self.index is None:
            print("[WARNING] Vector store is empty. No results to return.")
            return []

        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

    # ── NEW: Subject-filtered search ─────────────────────────────────────────

    def filtered_query(
        self,
        query_text: str,
        top_k: int = 8,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        Query the vector store with optional metadata filtering.

        Uses over-fetch + post-filter strategy:
          1. Retrieve top_k * multiplier results from FAISS (unfiltered).
          2. Post-filter by metadata key/value match.
          3. Return the top_k best-matching filtered results.

        Parameters
        ----------
        query_text : str
            The search query.
        top_k : int
            Number of filtered results to return.
        filter : dict, optional
            Metadata filter, e.g. {"subject": "DBMS"}.
            All key-value pairs must match (AND logic).
            If None, behaves identically to self.query().

        Returns
        -------
        list[dict]  Each dict has: index, distance, metadata.
        """
        if self.index is None:
            print("[WARNING] Vector store is empty.")
            return []

        if not filter:
            return self.query(query_text, top_k=top_k)

        # Over-fetch: request enough raw results that we're likely to find
        # top_k matching the filter even with skewed subject distribution.
        n_subjects = max(len(self.get_indexed_subjects()), 1)
        over_fetch_k = min(top_k * n_subjects * 3, self.index.ntotal)

        print(
            f"[INFO] Filtered query: '{query_text}' | "
            f"filter={filter} | over_fetch={over_fetch_k}"
        )

        query_emb = self.model.encode([query_text]).astype("float32")
        raw_results = self.search(query_emb, top_k=over_fetch_k)

        # Post-filter: keep only results where ALL filter keys match
        filtered = []
        for r in raw_results:
            meta = r.get("metadata")
            if not meta:
                continue
            if all(meta.get(k) == v for k, v in filter.items()):
                filtered.append(r)
            if len(filtered) >= top_k:
                break

        print(
            f"[INFO] Filtered results: {len(filtered)}/{len(raw_results)} "
            f"match filter {filter}"
        )
        return filtered

    # ── NEW: Subject introspection ────────────────────────────────────────────

    def get_indexed_subjects(self) -> Set[str]:
        """
        Return the set of unique subject values across all indexed metadata.
        Used by the classifier to know which subjects are available.
        """
        subjects = set()
        for meta in self.metadata:
            if isinstance(meta, dict) and "subject" in meta:
                subjects.add(meta["subject"])
        return subjects

    def get_index_stats(self) -> Dict:
        """Return summary statistics about the vector store."""
        subjects = {}
        for meta in self.metadata:
            if isinstance(meta, dict):
                subj = meta.get("subject", "_unknown")
                subjects[subj] = subjects.get(subj, 0) + 1
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_metadata": len(self.metadata),
            "subjects": subjects,
        }


if __name__ == "__main__":
    from src.data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print("Subjects:", store.get_indexed_subjects())
    print("Stats:", store.get_index_stats())
    print(store.query("What is Database Management System?", top_k=3))
