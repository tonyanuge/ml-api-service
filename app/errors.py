# app/errors.py

class DocuFlowError(Exception):
    """Base application error"""


class PermissionDenied(DocuFlowError):
    pass


class NoRouteMatched(DocuFlowError):
    pass


class EmptySearchResults(DocuFlowError):
    pass
