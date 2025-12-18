import os
import json
from datetime import datetime
from threading import Lock

import faiss
import numpy as np

from nlp.embedder import get_embedding


# =========================
# Configuration
# =========================

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "store.index")
META_PATH = os.path.join(DATA_DIR, "metadata.json")

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class FAISSStore:
    """
    Canonical FAISS store for DocuFlow v2.

    Guarantees:
    - Backward compatible with add_document(text)
    - Persistent across restarts
    - Thread-safe
    - FAISS is the single vector backend
    """

    _lock = Lock()

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

        # Load or create FAISS index
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            print("[FAISS] Loaded index from disk.")
        else:
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            faiss.write_index(self.index, INDEX_PATH)
            print("[FAISS] Created new index.")

        # Load or create metadata
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"[FAISS] Loaded {len(self.metadata)} metadata items.")
        else:
            self.metadata = []
            self._persist_metadata()
            print("[FAISS] Created new metadata store.")

    # =========================
    # Internal helpers
    # =========================

    def _persist_index(self):
        faiss.write_index(self.index, INDEX_PATH)

    def _persist_metadata(self):
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Normalises embedder output to float32 numpy array.
        Handles list or numpy outputs safely.
        """
        emb = get_embedding(text)

        # FIX: handle list output
        if isinstance(emb, list):
            emb = np.array(emb, dtype="float32")

        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        return emb.astype("float32")

    # =========================
    # Backward-compatible API
    # =========================

    def add_document(self, text_or_meta):
        """
        Backward-compatible ingestion method.

        Accepts:
        - str (existing behaviour)
        - dict with 'text' field (future metadata enrichment)

        This method is REQUIRED because main.py already calls it.
        """

        if isinstance(text_or_meta, str):
            text = text_or_meta
            meta = {"text": text}
        elif isinstance(text_or_meta, dict):
            text = text_or_meta.get("text")
            meta = text_or_meta
        else:
            raise ValueError("add_document expects str or dict")

        embedding = self._embed_text(text)

        with self._lock:
            self.index.add(embedding)

            meta.setdefault("created_at", datetime.utcnow().isoformat())
            meta.setdefault("vector_id", len(self.metadata))

            self.metadata.append(meta)

            self._persist_index()
            self._persist_metadata()

        print("[FAISS] Added 1 vector.")

    # =========================
    # Batch insert (future-safe)
    # =========================

    def add_embeddings(self, embeddings: np.ndarray, metadatas: list[dict]):
        """
        Batch insert for future ingestion pipelines.
        """

        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadata length mismatch")

        embeddings = embeddings.astype("float32")

        with self._lock:
            start_id = len(self.metadata)

            self.index.add(embeddings)

            for i, meta in enumerate(metadatas):
                meta.setdefault("created_at", datetime.utcnow().isoformat())
                meta.setdefault("vector_id", start_id + i)
                self.metadata.append(meta)

            self._persist_index()
            self._persist_metadata()

        print(f"[FAISS] Added {len(embeddings)} vectors.")

    # =========================
    # Search APIs
    # =========================

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Low-level FAISS search.
        """
        if self.index.ntotal == 0:
            return []

        D, I = self.index.search(
            query_embedding.astype("float32").reshape(1, -1),
            k
        )

        results = []
        for idx, dist in zip(I[0], D[0]):
            if 0 <= idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(dist)
                results.append(item)

        return results

    def search_by_text(self, query_text: str, k: int = 5):
        """
        Convenience wrapper: text -> embedding -> FAISS search.
        """
        emb = self._embed_text(query_text)
        return self.search(emb, k)

    def retrieve(self, query_embedding: np.ndarray, k: int = 10):
        """
        Retrieval layer for RAG-style pipelines.
        """
        return self.search(query_embedding, k)
