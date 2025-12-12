from fastapi import FastAPI
from pydantic import BaseModel
from nlp.preprocess import clean_text
from nlp.embedder import get_embedding, semantic_search

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

# 4 Preprocessing the text
    cleaned = clean_text(request.text)

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
    embedding = get_embedding(request.text)
    return {"embedding": embedding}


@app.post("/sementic-search")
def sementic_search_endpoint(request: SemanticSearchRequest):
    results = semantic_search(request.query, request.documents)
    return {"results": results}