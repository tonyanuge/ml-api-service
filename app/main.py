from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel

from nlp.preprocess import clean_text
from nlp.embedder import get_embedding
from hybrid_search import hybrid_search  # this MUST be a function
from workflow.router import WorkflowRouter
from vector_db.faiss_store import FAISSStore
from ingest_file.text_reader import read_text_file
from ingest_file.pdf_reader import extract_pdf_text
from ingest_file.docx_reader import extract_docx_text
from ingest_file.chunker import chunk_text
from workflow.executor import WorkflowExecutor
from security.guard import enforce_permission
from security.roles import Role, Capability


# ----------------------------------------------------
# Initialise FAISS once at startup
# ----------------------------------------------------
faiss_db = FAISSStore()
workflow_router = WorkflowRouter()
executor = WorkflowExecutor()
app = FastAPI()


# ----------------------------------------------------
# Request / Response Models
# ----------------------------------------------------
class TextRequest(BaseModel):
    text: str


class ClassificationResponse(BaseModel):
    label: str
    confidence: float


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = 10


class FAISSAddRequest(BaseModel):
    text: str


class FAISSSearchRequest(BaseModel):
    query: str
    top_k: int = 5


# ----------------------------------------------------
# ROOT
# ----------------------------------------------------
@app.get("/")
def root():
    return {"message": "ML API Service is running"}


# ----------------------------------------------------
# CLASSIFICATION
# ----------------------------------------------------
@app.post("/classify", response_model=ClassificationResponse)
def classify_text(request: TextRequest):

    cleaned = clean_text(request.text)

    if "urgent" in cleaned:
        return ClassificationResponse(label="urgent", confidence=0.92)
    elif "payment" in cleaned:
        return ClassificationResponse(label="payment_request", confidence=0.87)
    else:
        return ClassificationResponse(label="general", confidence=0.55)


# ----------------------------------------------------
# EMBEDDINGS
# ----------------------------------------------------
@app.post("/embed")
def embed_text(request: TextRequest):
    embedding = get_embedding(request.text)
    return {"embedding": embedding}


# ----------------------------------------------------
# SEMANTIC SEARCH (FAISS ONLY)
# ----------------------------------------------------
@app.post("/semantic-search")
def semantic_search_endpoint(
    request: SemanticSearchRequest = Body(...)
):
    results = faiss_db.search_by_text(
        query_text=request.query,
        k=request.top_k
    )
    return {"results": results}


# ----------------------------------------------------
# HYBRID SEARCH (FAISS → rerank)
# ----------------------------------------------------
@app.post("/hybrid-search")
def hybrid_search_endpoint(
    request: HybridSearchRequest = Body(...)
):
    # Step 1: semantic recall from FAISS
    semantic_results = faiss_db.search_by_text(
        query_text=request.query,
        k=request.top_k
    )

    # Step 2: hybrid reranking
    results = hybrid_search(
        query=request.query,
        faiss_store=faiss_db,
        top_k=request.top_k
    )


    return {"results": results}


# ----------------------------------------------------
# FAISS — ADD DOCUMENT
# ----------------------------------------------------
@app.post("/faiss/add")
def faiss_add(request: FAISSAddRequest):

    faiss_db.add_document(request.text)

    return {
        "status": "ok",
        "stored_text": request.text
    }


# ----------------------------------------------------
# FAISS — SEARCH
# ----------------------------------------------------
@app.post("/faiss/search")
def faiss_search(request: FAISSSearchRequest):

    results = faiss_db.search_by_text(
        query_text=request.query,
        k=request.top_k
    )

    return {"results": results}


# ----------------------------------------------------
# FILE INGESTION → FAISS
# ----------------------------------------------------
@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):

    ext = file.filename.lower()
    raw_bytes = await file.read()

    # Decide how to extract text
    if ext.endswith(".txt"):
        text = read_text_file(raw_bytes)
    elif ext.endswith(".pdf"):
        text = extract_pdf_text(raw_bytes)
    elif ext.endswith(".docx"):
        text = extract_docx_text(raw_bytes)
    else:
        return {"error": "Unsupported file type"}

    # Break into chunks
    chunks = chunk_text(text)

    # Store each chunk in FAISS
    for i, chunk in enumerate(chunks):
        faiss_db.add_document({
            "text": chunk,
            "source_file": file.filename,
            "chunk_id": i,
            "total_chunks": len(chunks),
            "pipeline": "api_upload"
        })

    return {
        "filename": file.filename,
        "chunks_stored": len(chunks)
    }



@app.post("/route-from-search")
def route_from_search(request: HybridSearchRequest):
    """
    End-to-end routing endpoint.

    Flow:
    1. Enforce permissions (Week 7.3)
    2. Run hybrid search (contextual only)
    3. Classify the query
    4. Route via YAML rules
    5. Execute routing decision
    6. Audit execution
    """

    # --- TEMP role assignment (Week 7.3) ---
    role = Role.OPERATOR

    # --- Permission enforcement ---
    enforce_permission(
        role=role,
        capability=Capability.ROUTE_WORKFLOW
    )

    # 1. Hybrid search
    search_results = hybrid_search(
        query=request.query,
        faiss_store=faiss_db,
        top_k=request.top_k
    )

    # 2. Classification
    classification_response = classify_text(
        TextRequest(text=request.query)
    )
    classification_label = classification_response.label

    # 3. YAML-based routing
    decision = workflow_router.route(
        classification=classification_label,
        text=request.query
    )

    # 4. Execute decision
    execution = executor.execute(
        decision=decision,
        context={
            "query": request.query,
            "classification": classification_label,
            "top_k": request.top_k
        }
    )

    return {
        "role": role,
        "query": request.query,
        "classification": classification_label,
        "decision": decision,
        "execution": execution,
        "search_results": search_results
    }
