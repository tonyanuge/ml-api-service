# workflow/executor.py

from audit.logger import AuditLogger

class WorkflowExecutor:
    def __init__(self):
        self.logger = AuditLogger()

    def execute(self, *, decision: dict, context: dict) -> dict:
        """
        Execute a routing decision.
        Returns execution result.
        """

        action = decision.get("action", "log")

        result = {
            "action": action,
            "status": "completed"
        }

        # ---- Supported actions ----

        if action == "queue":
            result["message"] = "Item queued for processing"

        elif action == "tag":
            result["message"] = "Metadata tag applied"

        elif action == "log":
            result["message"] = "Logged only (no side effects)"

        elif action == "webhook":
            result["message"] = "Webhook execution placeholder"

        else:
            result["status"] = "ignored"
            result["message"] = f"Unknown action: {action}"

        # ---- Audit everything ----
        self.logger.log(
            event="workflow_execution",
            payload={
                "decision": decision,
                "context": context,
                "result": result
            }
        )

        return result
