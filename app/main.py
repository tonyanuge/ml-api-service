from fastapi import FastAPI
from pydantic import BaseModel
from nlp.preprocess import clean_text

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