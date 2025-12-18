# workflow/router.py

import yaml
from pathlib import Path


# --------------------------------------------------
# Rules configuration
# --------------------------------------------------
RULES_PATH = Path(__file__).parent / "rules.yaml"


class WorkflowRouter:
    """
    YAML-driven workflow router.

    - Industry-agnostic
    - Deterministic
    - No side effects
    """

    def __init__(self, rules_path: Path = RULES_PATH):
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f) or {}

    def route(self, *, classification: str, text: str) -> dict:
        """
        Returns a routing decision based on rules.yaml.

        Matching order:
        1. classification (exact match)
        2. keyword_contains (any match)
        3. first rule wins
        4. fallback to default_route
        """

        text_lower = text.lower()

        for rule in self.rules.get("routes", []):
            conditions = rule.get("when", {})

            # --- Classification condition ---
            if "classification" in conditions:
                if classification != conditions["classification"]:
                    continue

            # --- Keyword condition ---
            if "keyword_contains" in conditions:
                keywords = conditions["keyword_contains"]
                if not any(k.lower() in text_lower for k in keywords):
                    continue

            # Match found
            return rule.get("route", {})

        # Fallback route
        return self.rules.get("default_route", {})
