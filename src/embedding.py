from typing import List, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.cleaner import get_default_cleaner

class EmbeddingPipeline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        self.cleaner = get_default_cleaner()
        print(f"[INFO] Loaded embedding model: {model_name}")
        
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        # 1. Clean documents first
        cleaned_docs = self.cleaner.clean_documents(documents)
        
        # 2. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " "]
        )
        
        chunks = splitter.split_documents(cleaned_docs)
        print(f"[INFO] Split {len(cleaned_docs)} documents into {len(chunks)} chunks.")
        return chunks
    
    def embed_chunks(self, chunks: List[Any], batch_size: int = 500) -> tuple[np.ndarray, List[Any]]:
        """
        Embeds chunks in batches for stability and better error reporting.
        Returns a tuple of (embeddings_array, valid_chunks_list).
        """
        valid_chunks = []
        all_embeddings = []
        
        print(f"[INFO] Generating embeddings for {len(chunks)} chunks in batches of {batch_size}...")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_texts = []
            current_batch_chunks = []
            
            # Additional validation per chunk
            for chunk in batch:
                content = getattr(chunk, 'page_content', None)
                if content is not None and isinstance(content, str) and content.strip():
                    batch_texts.append(content)
                    current_batch_chunks.append(chunk)
            
            if not batch_texts:
                continue
                
            try:
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                all_embeddings.append(batch_embeddings)
                valid_chunks.extend(current_batch_chunks)
            except Exception as e:
                print(f"[ERROR] Failed to embed batch starting at index {i}. Error: {e}")
                print("[INFO] Attempting to identify problematic chunk in batch...")
                for j, text in enumerate(batch_texts):
                    try:
                        self.model.encode([text], show_progress_bar=False)
                    except Exception as ex:
                        print(f"[ERROR] Problematic chunk found at original index {i+j}!")
                        # Safely encode for windows terminal printing to avoid crashes
                        safe_text = text[:100].encode('ascii', 'replace').decode('ascii')
                        print(f"[DEBUG] Content snippet: {safe_text}...")
                        # We skip this specific chunk and continue
                        continue
        
        if not all_embeddings:
            print("[WARNING] No embeddings were generated.")
            return np.array([]), []
            
        final_embeddings = np.vstack(all_embeddings)
        print(f"[INFO] Total valid embeddings: {final_embeddings.shape[0]} / {len(chunks)}")
        return final_embeddings, valid_chunks
    
if __name__ == "__main__":
    docs = load_all_documents('Research/data/pdf')
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print(f"[INFO] Example embeddings:", embeddings[0] if len(embeddings) > 0 else None)
        