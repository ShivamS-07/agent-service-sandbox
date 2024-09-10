import logging
from typing import List, Optional, Self

from grpclib import GRPCError
from stock_universe_service_proto_v1.custom_data_service_pb2 import (
    GetFileContentsResponse,
    GetFileInfoResponse,
    ListDocumentsResponse,
    ListingStatusType,
)

from agent_service.endpoints.models import (
    CustomDocumentListing,
    CustomDocumentSummaryChunk,
    GetCustomDocumentFileInfoResponse,
    GetCustomDocumentFileResponse,
    ListCustomDocumentsResponse,
)
from agent_service.external.custom_data_svc_client import (
    get_custom_doc_file_contents as get_custom_doc_file_contents_grpc,
)
from agent_service.external.custom_data_svc_client import (
    get_custom_doc_file_info as get_custom_doc_file_info_grpc,
)
from agent_service.external.custom_data_svc_client import list_custom_docs

LOGGER = logging.getLogger(__name__)


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


def document_listing_status_to_str(status: ListingStatusType) -> str:
    if status == ListingStatusType.LISTING_STATUS_PROCESSING:
        return "PROCESSING"
    if status == ListingStatusType.LISTING_STATUS_AVAILABLE:
        return "AVAILABLE"
    if status == ListingStatusType.LISTING_STATUS_ERROR:
        return "ERROR"
    return "UNKNOWN"


async def get_custom_doc_listings(
    user_id: str, base_path: str = "", recursive: bool = True
) -> ListCustomDocumentsResponse:
    try:
        resp: ListDocumentsResponse = await list_custom_docs(
            user_id=user_id, base_path=base_path, recursive=recursive
        )

        return ListCustomDocumentsResponse(
            documents=[
                CustomDocumentListing(
                    file_id=listing.file_id,
                    name=listing.name,
                    base_path=listing.base_path,
                    full_path=listing.full_path,
                    type=listing.type,
                    size=listing.size,
                    is_dir=listing.is_dir,
                    listing_status=document_listing_status_to_str(listing.listing_status.status),
                    upload_time=listing.upload_time.ToDatetime(),
                )
                for listing in resp.listings
            ],
        )
    except GRPCError as e:
        raise CustomDocumentException.from_grpc_error(e) from e


async def get_custom_doc_file(
    file_id: str, user_id: str, return_previewable_file: bool
) -> GetCustomDocumentFileResponse:
    try:
        resp: GetFileContentsResponse = await get_custom_doc_file_contents_grpc(
            user_id=user_id, file_id=file_id, return_previewable_file=return_previewable_file
        )

        return GetCustomDocumentFileResponse(
            is_preview=False,
            file_name=resp.file_name,
            file_type=resp.content_type,
            content=resp.raw_file,
        )
    except GRPCError as e:
        raise CustomDocumentException.from_grpc_error(e) from e


async def get_custom_doc_file_info(file_id: str, user_id: str) -> GetCustomDocumentFileInfoResponse:
    try:
        resp: GetFileInfoResponse = await get_custom_doc_file_info_grpc(
            user_id=user_id, file_id=file_id
        )

        # rpc is for a list of docs, try to grab the one we requested
        file_info = resp.file_info.get(file_id)
        if file_info is None:
            raise CustomDocumentException(
                f"document not found for file_id: {file_id}", ["No file info found"]
            )

        return GetCustomDocumentFileInfoResponse(
            file_id=file_info.file_id,
            author=file_info.author,
            status=document_listing_status_to_str(file_info.status),
            file_type=file_info.file_type,
            file_size=file_info.size,
            author_org=file_info.author_org,
            upload_time=file_info.upload_time.ToDatetime(),
            publication_time=file_info.publication_time.ToDatetime(),
            company_name=file_info.company_name,
            spiq_company_id=file_info.spiq_company_id,
            file_paths=[f for f in file_info.file_paths],
            chunks=[
                CustomDocumentSummaryChunk(
                    chunk_id=r.chunk_id,
                    headline=r.headline,
                    summary=r.summary,
                    long_summary=r.long_summary,
                )
                for r in file_info.chunks
            ],
        )
    except GRPCError as e:
        raise CustomDocumentException.from_grpc_error(e) from e
