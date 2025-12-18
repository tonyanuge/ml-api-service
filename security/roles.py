# security/roles.py

from enum import Enum
from typing import Set, Dict


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
# Role → Capability mapping
# --------------------------------------------------

ROLE_CAPABILITIES: Dict[Role, Set[Capability]] = {
    Role.VIEWER: {
        Capability.VIEW_SEARCH_RESULTS,
    },

    Role.OPERATOR: {
        Capability.VIEW_SEARCH_RESULTS,
        Capability.EXECUTE_WORKFLOW,
    },

    Role.MANAGER: {
        Capability.VIEW_SEARCH_RESULTS,
        Capability.EXECUTE_WORKFLOW,
        Capability.OVERRIDE_ROUTE,
        Capability.VIEW_AUDIT_LOGS,
    },

    Role.ADMIN: {
        Capability.VIEW_SEARCH_RESULTS,
        Capability.EXECUTE_WORKFLOW,
        Capability.OVERRIDE_ROUTE,
        Capability.VIEW_AUDIT_LOGS,
        Capability.MANAGE_RULES,
    },
}


# --------------------------------------------------
# Helper utilities (used later by guards)
# --------------------------------------------------

def role_has_capability(role: Role, capability: Capability) -> bool:
    """
    Check if a role allows a given capability.
    """
    return capability in ROLE_CAPABILITIES.get(role, set())
