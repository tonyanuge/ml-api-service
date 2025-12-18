# app/config/settings.py

import os
from pathlib import Path
from security.roles import Role


# -----------------------------
# Environment
# -----------------------------
ENVIRONMENT = os.getenv("DOCUFLOW_ENV", "local")


# -----------------------------
# Default role (temporary)
# -----------------------------
DEFAULT_ROLE = Role(
    os.getenv("DOCUFLOW_DEFAULT_ROLE", Role.OPERATOR.value)
)


# -----------------------------
# Audit logging
# -----------------------------
AUDIT_LOG_DIR = os.getenv("DOCUFLOW_AUDIT_LOG_DIR", "audit_logs")
AUDIT_LOG_FILE = os.getenv(
    "DOCUFLOW_AUDIT_LOG_FILE",
    "workflow_audit.jsonl"
)


# -----------------------------
# FAISS storage
# -----------------------------
FAISS_DATA_DIR = Path(
    os.getenv("DOCUFLOW_FAISS_DATA_DIR", "data")
)

FAISS_INDEX_FILE = os.getenv(
    "DOCUFLOW_FAISS_INDEX_FILE",
    "store.index"
)

FAISS_METADATA_FILE = os.getenv(
    "DOCUFLOW_FAISS_METADATA_FILE",
    "metadata.json"
)
