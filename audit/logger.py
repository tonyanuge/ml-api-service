# audit/logger.py

from datetime import datetime
from pathlib import Path
import json

from app.config.settings import AUDIT_LOG_DIR, AUDIT_LOG_FILE


# --------------------------------------------------
# Audit log directory (config-driven)
# --------------------------------------------------

AUDIT_DIR = Path(AUDIT_LOG_DIR)
AUDIT_DIR.mkdir(exist_ok=True)


class AuditLogger:
    """
    Append-only audit logger.

    Writes JSONL records for all governed actions.
    """

    def __init__(self, log_file: str = AUDIT_LOG_FILE):
        self.log_path = AUDIT_DIR / log_file

    def log(self, *, event: str, payload: dict):
        record = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
