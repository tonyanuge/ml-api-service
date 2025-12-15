import re
import numpy as np
from nlp.embedder import get_embedding

print("[HYBRID] Loaded hybrid_search.py from:", __file__)


def keyword_score(query: str, doc: str) -> float:
    query_tokens = set(re.findall(r"\w+", query.lower()))
    doc_tokens = set(re.findall(r"\w+", doc.lower()))

    if not query_tokens:
        return 0.0

    overlap = query_tokens.intersection(doc_tokens)
    return len(overlap) / len(query_tokens)


def semantic_cosine(a, b):
    """Simple cosine similarity for 1D vectors"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_score(query: str, doc: str) -> float:
    q_emb = get_embedding(query)
    d_emb = get_embedding(doc)

    print("[HYBRID] q_emb shape:", len(q_emb))
    print("[HYBRID] d_emb shape:", len(d_emb))

    return semantic_cosine(q_emb, d_emb)


def hybrid_search(query: str, documents: list[str]):
    print("[HYBRID] Running hybrid search on", len(documents), "documents")

    results = []

    for doc in documents:
        k = keyword_score(query, doc)
        s = semantic_score(query, doc)

        combined = 0.3 * k + 0.7 * s

        results.append({
            "document": doc,
            "keyword_score": k,
            "semantic_score": s,
            "combined_score": combined
        })

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results
