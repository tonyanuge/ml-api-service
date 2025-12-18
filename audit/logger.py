# audit/logger.py

from datetime import datetime
from pathlib import Path
import json

AUDIT_DIR = Path("audit_logs")
AUDIT_DIR.mkdir(exist_ok=True)

class AuditLogger:
    def __init__(self, log_file: str = "workflow_audit.jsonl"):
        self.log_path = AUDIT_DIR / log_file

    def log(self, *, event: str, payload: dict):
        record = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
