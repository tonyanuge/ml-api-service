from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str):
    """Return a vector embedding for the input text."""
    print(f"[DEBUG] get_embedding input: {text[:50]}...")
    if not text or not isinstance(text, str):
        return []

    embedding = model.encode([text])[0]
    print(f"[DEBUG] get_embedding length: {len(embedding)}")
    return embedding.tolist()


# -------------------------------------------------------
# NEW: Pure Python cosine similarity for 1D vectors
# -------------------------------------------------------
def cosine_sim_1d(vec_a, vec_b):
    """Cosine similarity for 1D numpy arrays or lists."""
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)

    if norm == 0:
        return 0.0

    return float(dot / norm)


def semantic_search(query: str, documents: list):
    """Semantic search using sklearn cosine similarity."""
    print(f"[DEBUG] semantic_search called.")
    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] Documents: {documents}")

    if not documents or not isinstance(documents, list):
        print("[DEBUG] Invalid or empty document list")
        return []

    query_emb = np.array(get_embedding(query)).reshape(1, -1)
    doc_embs = np.array([get_embedding(d) for d in documents])

    scores = cosine_similarity(query_emb, doc_embs)[0]
    print(f"[DEBUG] Similarity scores: {scores}")

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"[DEBUG] Ranked results: {ranked}")

    return [
        {"document": doc, "score": float(score)}
        for doc, score in ranked
    ]
