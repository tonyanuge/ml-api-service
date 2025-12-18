# --------------------------------------------------
# DocuFlow v2 – Backend Container
# --------------------------------------------------

FROM python:3.11-slim

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Workdir
# -----------------------------
WORKDIR /app

# -----------------------------
# Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Application code
# -----------------------------
COPY . .

# -----------------------------
# Environment defaults
# (can be overridden by docker-compose)
# -----------------------------
ENV DOCUFLOW_ENV=container
ENV DOCUFLOW_DEFAULT_ROLE=operator
ENV DOCUFLOW_AUDIT_LOG_DIR=/app/audit_logs
ENV DOCUFLOW_FAISS_DATA_DIR=/app/data

# -----------------------------
# Expose API
# -----------------------------
EXPOSE 8001

# -----------------------------
# Run FastAPI
# -----------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
