from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from nlp.preprocess import clean_text
from nlp.embedder import get_embedding
from hybrid_search import hybrid_search  # MUST be a function
from fastapi.middleware.cors import CORSMiddleware

from vector_db.faiss_store import FAISSStore
from workflow.router import WorkflowRouter
from workflow.executor import WorkflowExecutor
from security.guard import enforce_permission
from security.roles import Role, Capability
from audit.logger import AuditLogger

from ingest_file.text_reader import read_text_file
from ingest_file.pdf_reader import extract_pdf_text
from ingest_file.docx_reader import extract_docx_text
from ingest_file.chunker import chunk_text
from app.errors import PermissionDenied, NoRouteMatched, EmptySearchResults
from app.errors import DocuFlowError

from audit.events import (
    ROUTE_DECISION,
    ROUTE_EXECUTED,
    ROUTE_DENIED,
    ROUTE_FAILED,
)


# ----------------------------------------------------
# App + Core Services
# ----------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # OK for local + demo
    allow_credentials=True,
    allow_methods=["*"],        # Enables OPTIONS
    allow_headers=["*"],
)

faiss_db = FAISSStore()
workflow_router = WorkflowRouter()
executor = WorkflowExecutor()
audit_logger = AuditLogger()

# ----------------------------------------------------
# UI (Static HTML)
# ----------------------------------------------------
UI_DIR = Path(__file__).resolve().parent.parent / "ui"

app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


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



@app.get("/ui")
def serve_ui():
    return FileResponse(UI_DIR / "index.html")

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

    if ext.endswith(".txt"):
        text = read_text_file(raw_bytes)
    elif ext.endswith(".pdf"):
        text = extract_pdf_text(raw_bytes)
    elif ext.endswith(".docx"):
        text = extract_docx_text(raw_bytes)
    else:
        return {"error": "Unsupported file type"}

    chunks = chunk_text(text)

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


# ----------------------------------------------------
# ROUTE FROM SEARCH (Governed, Audited)
# ----------------------------------------------------
@app.post("/route-from-search")
def route_from_search(request: HybridSearchRequest):

    role = Role.OPERATOR

    try:
        # 1. Permission check
        enforce_permission(
            role=role,
            capability=Capability.EXECUTE_WORKFLOW
        )

        # 2. Hybrid search
        search_results = hybrid_search(
            query=request.query,
            faiss_store=faiss_db,
            top_k=request.top_k
        )

        if not search_results:
            raise EmptySearchResults("No relevant documents found")

        # 3. Classification
        classification_response = classify_text(
            TextRequest(text=request.query)
        )
        classification_label = classification_response.label

        # 4. Routing decision
        decision = workflow_router.route(
            classification=classification_label,
            text=request.query
        )

        if not decision:
            raise NoRouteMatched("No workflow rule matched")

        # --- AUDIT: decision made ---
        audit_logger.log(
            event=ROUTE_DECISION,
            payload={
                "role": role,
                "query": request.query,
                "classification": classification_label,
                "decision": decision,
            }
        )

        # 5. Execute
        execution = executor.execute(
            decision=decision,
            context={
                "query": request.query,
                "classification": classification_label,
                "top_k": request.top_k
            }
        )

        # --- AUDIT: execution completed ---
        audit_logger.log(
            event=ROUTE_EXECUTED,
            payload={
                "role": role,
                "query": request.query,
                "classification": classification_label,
                "decision": decision,
                "execution": execution,
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

    except PermissionDenied as e:
        audit_logger.log(
            event=ROUTE_DENIED,
            payload={
                "role": role,
                "query": request.query,
                "reason": str(e)
            }
        )
        raise

    except DocuFlowError as e:
        audit_logger.log(
            event=ROUTE_FAILED,
            payload={
                "role": role,
                "query": request.query,
                "error": str(e)
            }
        )
        raise
