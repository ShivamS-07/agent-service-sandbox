import datetime
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Optional, Tuple

from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)
from grpclib.client import Channel
from stock_universe_service_proto_v1.custom_data_service_grpc import (
    CustomDataServiceStub,
)
from stock_universe_service_proto_v1.custom_data_service_pb2 import (
    CheckDocumentUploadQuotaRequest,
    CheckDocumentUploadQuotaResponse,
    GetCitationContextRequest,
    GetCitationContextResponse,
    GetDocsByFileNamesRequest,
    GetDocsByFileNamesResponse,
    GetDocsBySecurityRequest,
    GetDocsBySecurityResponse,
    GetFileChunkInfoRequest,
    GetFileChunkInfoResponse,
    GetFileContentsRequest,
    GetFileContentsResponse,
    GetFileInfoRequest,
    GetFileInfoResponse,
    ListDocumentsRequest,
    ListDocumentsResponse,
    ListingStatusType,
    SemanticSearchDocsRequest,
    SemanticSearchDocsResponse,
)

from agent_service.external.grpc_utils import (
    date_to_timestamp,
    get_default_grpc_metadata,
    grpc_retry,
)
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("custom-data-service-dev.boosted.ai", 50051),
    DEV_TAG: ("custom-data-service-dev.boosted.ai", 50051),
    PROD_TAG: ("custom-data-service-prod.boosted.ai", 50051),
}


def document_listing_status_to_str(status: ListingStatusType) -> str:
    if status == ListingStatusType.LISTING_STATUS_PROCESSING:
        return "PROCESSING"
    if status == ListingStatusType.LISTING_STATUS_AVAILABLE:
        return "AVAILABLE"
    if status == ListingStatusType.LISTING_STATUS_ERROR:
        return "ERROR"
    return "UNKNOWN"


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("CUSTOM_DATA_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found CUSTOM_DATA_SERVICE_URL override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[CustomDataServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        channel = Channel(url, port)
        yield CustomDataServiceStub(channel)
    finally:
        channel.close()


@grpc_retry
@async_perf_logger
async def get_custom_docs_by_security(
    user_id: str,
    gbi_ids: List[int],
    limit: Optional[int] = None,
    publish_date_start: Optional[datetime.date] = None,
    publish_date_end: Optional[datetime.date] = None,
) -> GetDocsBySecurityResponse:
    with _get_service_stub() as stub:
        req = GetDocsBySecurityRequest(
            gbi_ids=gbi_ids,
            limit=limit,
            from_publication_time=date_to_timestamp(publish_date_start),
            to_publication_time=date_to_timestamp(publish_date_end),
            version="v2",
        )
        resp: GetDocsBySecurityResponse = await stub.GetDocsBySecurity(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def get_custom_docs_by_topic(
    user_id: str,
    topic: str,
    limit: Optional[int] = None,
    publish_date_start: Optional[datetime.date] = None,
    publish_date_end: Optional[datetime.date] = None,
) -> SemanticSearchDocsResponse:
    with _get_service_stub() as stub:
        req = SemanticSearchDocsRequest(
            query=topic,
            limit=limit,
            from_publication_time=date_to_timestamp(publish_date_start),
            to_publication_time=date_to_timestamp(publish_date_end),
            version="v2",
        )
        resp: SemanticSearchDocsResponse = await stub.SemanticSearchDocs(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def get_custom_docs_by_file_names(
    user_id: str,
    file_names: List[str],
) -> GetDocsByFileNamesResponse:
    with _get_service_stub() as stub:
        req = GetDocsByFileNamesRequest(
            file_names=file_names,
            version="v2",
        )
        resp: GetDocsByFileNamesResponse = await stub.GetDocsByFileNames(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def get_custom_doc_articles_info(
    user_id: str,
    article_ids: List[str],
) -> GetFileChunkInfoResponse:
    with _get_service_stub() as stub:
        req = GetFileChunkInfoRequest(
            file_chunk_ids=article_ids,
            version="v2",
        )
        resp: GetFileChunkInfoResponse = await stub.GetFileChunkInfo(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def get_custom_doc_file_info(user_id: str, file_id: str) -> GetFileInfoResponse:
    with _get_service_stub() as stub:
        req = GetFileInfoRequest(file_ids=[file_id])
        resp: GetFileInfoResponse = await stub.GetFileInfo(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def get_custom_doc_file_contents(
    user_id: str, file_id: str, return_previewable_file: bool = False
) -> GetFileContentsResponse:
    with _get_service_stub() as stub:
        req = GetFileContentsRequest(
            file_id=file_id, return_previewable_file=return_previewable_file
        )
        resp: GetFileContentsResponse = await stub.GetFileContents(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def check_custom_doc_upload_quota(
    user_id: str, candidate_total_size: int
) -> CheckDocumentUploadQuotaResponse:
    with _get_service_stub() as stub:
        req = CheckDocumentUploadQuotaRequest(request_total_file_size=candidate_total_size)
        resp: CheckDocumentUploadQuotaResponse = await stub.CheckDocumentUploadQuota(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def list_custom_docs(
    user_id: str, base_path: str = "", recursive: bool = True
) -> ListDocumentsResponse:
    with _get_service_stub() as stub:
        req = ListDocumentsRequest(base_path=base_path, recursive=recursive)
        resp: ListDocumentsResponse = await stub.ListDocuments(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def get_citation_custom_doc_context(
    user_id: str, citation_id: str
) -> GetCitationContextResponse:
    with _get_service_stub() as stub:
        req = GetCitationContextRequest(citation_id=citation_id)
        resp: GetCitationContextResponse = await stub.GetCitationContext(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp
