import json
from datetime import datetime
from threading import Lock
from pathlib import Path

import faiss
import numpy as np

from nlp.embedder import get_embedding
from app.config.settings import (
    FAISS_DATA_DIR,
    FAISS_INDEX_FILE,
    FAISS_METADATA_FILE
)

# =========================
# Configuration (externalised)
# =========================

DATA_DIR: Path = FAISS_DATA_DIR
INDEX_PATH: Path = DATA_DIR / FAISS_INDEX_FILE
META_PATH: Path = DATA_DIR / FAISS_METADATA_FILE

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
        DATA_DIR.mkdir(exist_ok=True)

        # Load or create FAISS index
        if INDEX_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            print("[FAISS] Loaded index from disk.")
        else:
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            faiss.write_index(self.index, str(INDEX_PATH))
            print("[FAISS] Created new index.")

        # Load or create metadata
        if META_PATH.exists():
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
        faiss.write_index(self.index, str(INDEX_PATH))

    def _persist_metadata(self):
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Normalises embedder output to float32 numpy array.
        Handles list or numpy outputs safely.
        """
        emb = get_embedding(text)

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
        - str
        - dict with 'text' field
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
        emb = self._embed_text(query_text)
        return self.search(emb, k)

    def retrieve(self, query_embedding: np.ndarray, k: int = 10):
        return self.search(query_embedding, k)
