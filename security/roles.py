# security/roles.py

from enum import Enum
from typing import Set, Dict
from pathlib import Path
import yaml


class Role(str, Enum):
    """
    System roles.

    These are intentionally generic and industry-agnostic.
    """
    VIEWER = "viewer"
    OPERATOR = "operator"
    MANAGER = "manager"
    ADMIN = "admin"


class Capability(str, Enum):
    """
    Atomic actions the system understands.

    These map to things the system can DO,
    not UI concepts.
    """
    VIEW_SEARCH_RESULTS = "view_search_results"
    EXECUTE_WORKFLOW = "execute_workflow"
    OVERRIDE_ROUTE = "override_route"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_RULES = "manage_rules"


# --------------------------------------------------
# Load Role → Capability mapping from YAML
# --------------------------------------------------

_PERMISSIONS_PATH = Path(__file__).parent / "permissions.yaml"

with open(_PERMISSIONS_PATH, "r", encoding="utf-8") as f:
    _permissions_config = yaml.safe_load(f)


ROLE_CAPABILITIES: Dict[Role, Set[Capability]] = {}

for role_value, caps in _permissions_config.get("roles", {}).items():
    role = Role(role_value)
    ROLE_CAPABILITIES[role] = {Capability(c) for c in caps}


# --------------------------------------------------
# Helper utilities (used by guards)
# --------------------------------------------------

def role_has_capability(role: Role, capability: Capability) -> bool:
    """
    Check if a role allows a given capability.
    """
    return capability in ROLE_CAPABILITIES.get(role, set())
