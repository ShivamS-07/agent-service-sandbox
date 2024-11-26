import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile
from starlette import status

from agent_service.endpoints.authz_helper import User, parse_header
from agent_service.endpoints.models import (
    AddCustomDocumentsResponse,
    CheckCustomDocumentUploadQuotaResponse,
    DeleteCustomDocumentsRequest,
    DeleteCustomDocumentsResponse,
    GetCustomDocumentFileInfoResponse,
    ListCustomDocumentsResponse,
)
from agent_service.endpoints.routers.utils import get_agent_svc_impl
from agent_service.utils.custom_documents_utils import (
    CustomDocumentException,
    CustomDocumentQuotaExceededException,
)

router = APIRouter(prefix="/api/custom-documents")
logger = logging.getLogger(__name__)


# custom doc endpoints
@router.get("", response_model=ListCustomDocumentsResponse, status_code=status.HTTP_200_OK)
async def list_custom_docs(user: User = Depends(parse_header)) -> ListCustomDocumentsResponse:
    """
    Gets custom document file content as a byte stream
    Args:
        file_id (str): the file's ID
    """
    agent_svc_impl = get_agent_svc_impl()
    try:
        return await agent_svc_impl.list_custom_documents(user=user)
    except CustomDocumentException as e:
        logger.exception("Error while listing custom docs")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.message)


@router.get(
    "/{file_id}/download",
    response_class=Response,
    status_code=status.HTTP_200_OK,
)
async def get_custom_doc_file(
    file_id: str, preview: bool = False, user: User = Depends(parse_header)
) -> Response:
    """
    Gets custom document file content as a byte stream
    Args:
        file_id (str): the file's ID
    Query Params:
        preview (bool): whether to return a previewable version of the file
                        ie: a PDF for files more complex than txt.
    """
    agent_svc_impl = get_agent_svc_impl()
    try:
        resp = await agent_svc_impl.get_custom_doc_file_content(
            user=user, file_id=file_id, return_previewable_file=preview
        )
        return Response(content=resp.content, media_type=resp.file_type)
    except CustomDocumentException as e:
        logger.exception(f"Error while downloading custom doc {file_id}; {preview=}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.message
        ) from e


@router.get(
    "/{file_id}/info",
    response_model=GetCustomDocumentFileInfoResponse,
    status_code=status.HTTP_200_OK,
)
async def get_custom_doc_details(
    file_id: str, user: User = Depends(parse_header)
) -> GetCustomDocumentFileInfoResponse:
    """
    Gets custom document details
    Args:
        file_id (str): the file's ID
    """
    agent_svc_impl = get_agent_svc_impl()
    try:
        return await agent_svc_impl.get_custom_doc_file_info(user=user, file_id=file_id)
    except CustomDocumentException as e:
        logger.exception(f"Error while getting custom doc metadata {file_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.message)


@router.post(
    "/add-documents",
    response_model=AddCustomDocumentsResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def add_custom_docs(
    files: list[UploadFile], base_path: Optional[str] = "", user: User = Depends(parse_header)
) -> AddCustomDocumentsResponse:
    """
    Uploads custom documents; will overwrite/reprocess existing files if uploaded again
    Args:
        body should be multipart/form-data with
            - `files` key containing the file(s) to upload
            - `base_path` (optional) key containing the base path (directory) to upload the files to
                          when omitted (default behaviour), files are uploaded to the root path for the user
    """
    agent_svc_impl = get_agent_svc_impl()
    try:
        return await agent_svc_impl.add_custom_documents(
            user=user, files=files, base_path=base_path, allow_overwrite=True
        )
    except CustomDocumentQuotaExceededException as e:
        logger.warning(
            f"User {user.user_id} attempted to upload custom documents over quota: {e.message}"
        )
        raise HTTPException(status_code=status.HTTP_507_INSUFFICIENT_STORAGE, detail=e.message)


@router.post(
    "/delete-documents",
    response_model=DeleteCustomDocumentsResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_custom_docs(
    req: DeleteCustomDocumentsRequest, user: User = Depends(parse_header)
) -> DeleteCustomDocumentsResponse:
    """
    Deletes custom documents
    """
    agent_svc_impl = get_agent_svc_impl()
    if req.file_paths is None or len(req.file_paths) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file paths provided"
        )
    resp = await agent_svc_impl.delete_custom_documents(user=user, file_paths=req.file_paths)
    return resp


@router.get(
    "/quota",
    response_model=CheckCustomDocumentUploadQuotaResponse,
    status_code=status.HTTP_200_OK,
)
async def get_custom_doc_quota(
    candidate_total_size: Optional[int] = 0, user: User = Depends(parse_header)
) -> CheckCustomDocumentUploadQuotaResponse:
    """
    Gets the available custom document upload quota for the user and checks if they have capacity
    for the candidate size of file(s) provided
    """
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.check_document_upload_quota(
        user=user, candidate_total_size=candidate_total_size
    )
