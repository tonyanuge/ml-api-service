from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    """Return a vector embedding for the input text."""
    if not text or not isinstance(text, str):
        return []

    # Correct extraction
    embedding = model.encode([text])[0]
    return embedding.tolist()


def semantic_search(query: str, documents: list):
    """
    Performs semantic search by comparing query embedding to document embeddings.
    Returns ranked documents with similarity scores.
    """
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

    # Rank results: highest score first
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
