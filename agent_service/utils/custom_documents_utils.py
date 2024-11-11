from typing import List, Optional, Self

from grpclib import GRPCError
from stock_universe_service_proto_v1.custom_data_service_pb2 import (
    CustomDocumentListing as ProtoCustomDocumentListing,
)

from agent_service.endpoints.models import CustomDocumentListing
from agent_service.external.custom_data_svc_client import document_listing_status_to_str
from agent_service.external.grpc_utils import timestamp_to_datetime


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


class CustomDocumentQuotaExceededException(CustomDocumentException):
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message, errors)


def custom_doc_listing_proto_to_model(
    listing: ProtoCustomDocumentListing,
) -> CustomDocumentListing:
    listing = CustomDocumentListing(
        file_id=listing.file_id,
        name=listing.name,
        base_path=listing.base_path,
        full_path=listing.full_path,
        type=listing.type,
        size=listing.size,
        is_dir=listing.is_dir,
        listing_status=document_listing_status_to_str(listing.listing_status.status),
        upload_time=timestamp_to_datetime(listing.upload_time),
        publication_time=timestamp_to_datetime(listing.publication_time),
        author=listing.author,
        author_org=listing.author_org,
        company_name=listing.company_name,
        spiq_company_id=listing.spiq_company_id,
    )
    return listing
