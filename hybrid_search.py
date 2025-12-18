import re
import numpy as np
from typing import List, Dict

from nlp.embedder import get_embedding

print("[HYBRID] Loaded hybrid_search.py from:", __file__)


# ====================================================
# Keyword utilities
# ====================================================

def keyword_score(query: str, text: str) -> float:
    """
    Simple keyword overlap score.
    """
    query_tokens = set(re.findall(r"\w+", query.lower()))
    text_tokens = set(re.findall(r"\w+", text.lower()))

    if not query_tokens:
        return 0.0

    overlap = query_tokens.intersection(text_tokens)
    return len(overlap) / len(query_tokens)


# ====================================================
# FAISS-aware reranker
# ====================================================

def rerank(query: str, semantic_results: List[Dict]) -> List[Dict]:
    """
    Rerank FAISS results using keyword overlap + semantic distance.

    semantic_results items MUST contain:
    - text
    - score  (FAISS L2 distance; lower is better)
    """

    print("[HYBRID] Reranking", len(semantic_results), "FAISS results")

    results = []

    for item in semantic_results:
        text = item.get("text", "")

        k_score = keyword_score(query, text)

        # Convert FAISS distance → similarity
        semantic_sim = 1 / (1 + item.get("score", 0.0))

        combined = 0.3 * k_score + 0.7 * semantic_sim

        enriched = item.copy()
        enriched["keyword_score"] = k_score
        enriched["combined_score"] = combined

        results.append(enriched)

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results


# ====================================================
# Step 5.3 – Canonical Hybrid Search (FAISS-first)
# ====================================================

def hybrid_search(
    query: str,
    faiss_store,
    top_k: int = 10,
    max_chunks_per_doc: int = 3
) -> List[Dict]:
    """
    Production hybrid search.

    Pipeline:
    1. Embed query
    2. Retrieve top-k chunks from FAISS
    3. Rerank using keyword + semantic score
    4. Group results by source_file
    5. Return document-level results

    Returns:
    [
        {
            "source_file": "...",
            "score": float,
            "chunks": [str, str, ...]
        }
    ]
    """

    print("[HYBRID] Running grouped hybrid search")

    # ------------------------------------------------
    # Embed query
    # ------------------------------------------------
    query_embedding = faiss_store._embed_text(query)

    # ------------------------------------------------
    # Retrieve from FAISS (chunk-level)
    # ------------------------------------------------
    raw_results = faiss_store.search(query_embedding, top_k)


    if not raw_results:
        return []

    # ------------------------------------------------
    # Rerank chunk-level results
    # ------------------------------------------------
    reranked = rerank(query, raw_results)

    # ------------------------------------------------
    # Group by source document
    # ------------------------------------------------
    grouped = {}

    for item in reranked:
        source = item.get("source_file", "unknown")

        grouped.setdefault(source, {
            "source_file": source,
            "best_score": item["combined_score"],
            "chunks": []
        })

        grouped[source]["chunks"].append(item)
        grouped[source]["best_score"] = max(
            grouped[source]["best_score"],
            item["combined_score"]
        )

    # ------------------------------------------------
    # Limit chunks per document
    # ------------------------------------------------
    for doc in grouped.values():
        doc["chunks"] = sorted(
            doc["chunks"],
            key=lambda x: x["combined_score"],
            reverse=True
        )[:max_chunks_per_doc]

    # ------------------------------------------------
    # Order documents by best score
    # ------------------------------------------------
    ordered_docs = sorted(
        grouped.values(),
        key=lambda x: x["best_score"],
        reverse=True
    )

    # ------------------------------------------------
    # Clean response
    # ------------------------------------------------
    response = []
    for doc in ordered_docs:
        response.append({
            "source_file": doc["source_file"],
            "score": doc["best_score"],
            "chunks": [c["text"] for c in doc["chunks"]]
        })

    return response
