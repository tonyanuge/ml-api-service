import yaml
from pathlib import Path


RULES_PATH = Path(__file__).parent / "rules.yaml"


class WorkflowRouter:
    def __init__(self, rules_path: Path = RULES_PATH):
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = yaml.safe_load(f)

    def route(self, *, classification: str, text: str) -> dict:
        """
        Returns routing decision based on rules.yaml
        """

        text_lower = text.lower()

        for rule in self.rules.get("routes", []):
            conditions = rule.get("when", {})

            # Classification match
            if "classification" in conditions:
                if classification != conditions["classification"]:
                    continue

            # Keyword match
            if "keyword_contains" in conditions:
                keywords = conditions["keyword_contains"]
                if not any(k in text_lower for k in keywords):
                    continue

            # Match found
            return rule["route"]

        # Fallback
        return self.rules.get("default_route", {})
