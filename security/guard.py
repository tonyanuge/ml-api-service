# security/guard.py

from security.roles import Role, Capability, role_has_capability


class PermissionDenied(Exception):
    """
    Raised when a role attempts an action it is not allowed to perform.
    """
    pass


def enforce_permission(
    *,
    role: Role,
    capability: Capability
) -> None:
    """
    Enforce that a role has a given capability.

    Raises PermissionDenied if not allowed.
    """

    if not role_has_capability(role, capability):
        raise PermissionDenied(
            f"Role '{role}' is not allowed to perform '{capability}'"
        )
