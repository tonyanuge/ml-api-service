from fastapi import FastAPI
from pydantic import BaseModel
from nlp.preprocess import clean_text
from nlp.embedder import get_embedding, semantic_search
import hybrid_search


app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {"message": "ML API Service is running"}

# 1. Define request body
class TextRequest(BaseModel):
    text: str


# 2. Define response body
class ClassificationResponse(BaseModel):
    label: str
    confidence: float

class SemanticSearchRequest(BaseModel):
    query: str
    documents: list[str]

# 3. Calssified API endpoint
@app.post("/classify", response_model=ClassificationResponse)
def classify_text(request: TextRequest):
    print(f"[DEBUG] /classify input: {request.text}")

# 4 Preprocessing the text
    cleaned = clean_text(request.text)
    print(f"[DEBUG] cleaned: {cleaned}")

# 5. Basic classification logic using cleaned text
    if "urgent" in cleaned:
        return ClassificationResponse(label="urgent", confidence=0.92)
    elif "paymeny" in cleaned:
        return ClassificationResponse(label="payment_request", confidence=0.87)
    else:
        return ClassificationResponse(label="general", confidence=0.55)
#   
@app.post("/embed")
def embed_text(request: TextRequest):
    print(f"[DEBUG] /embed input: {request.text}")
    embedding = get_embedding(request.text)
    print(f"[DEBUG] embedding length: {len(embedding)}")
    return {"embedding": embedding}


@app.post("/semantic-search")
def sementic_search_endpoint(request: SemanticSearchRequest):
    print(f"[DEBUG] /semantic-search query: {request.query}")
    print(f"[DEBUG] /semantic-search documents: {len(request.documents)} items")
    
    results = semantic_search(request.query, request.documents)

    print(f"[DEBUG] /semantic-search results: {results}")
    return {"results": results}


class HybridSearchRequest(BaseModel):
    query: str
    documents: list[str]


@app.post("/hybrid-search")
def hybrid_search_endpoint(request: HybridSearchRequest):
    print("[DEBUG] /hybrid-search query:", request.query)
    print("[DEBUG] /hybrid-search docs:", len(request.documents))

    results = hybrid_search.hybrid_search(request.query, request.documents)

    print("[DEBUG] /hybrid-search results:", results[:2])  # show top 2
    return {"results": results}
