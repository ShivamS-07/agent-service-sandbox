from typing import List, Optional, Self

from grpclib import GRPCError


class CustomDocumentException(Exception):
    message: str
    reason: str
    errors: Optional[List[str]]

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors

    @classmethod
    def from_grpc_error(cls, error: GRPCError) -> Self:
        e = cls(
            error.message if error.message else str(error),
            [str(detail) for detail in error.details] if error.details else None,
        )
        e.reason = error.status.name
        return e
