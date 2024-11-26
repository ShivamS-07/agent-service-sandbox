from fastapi import APIRouter, Depends
from starlette import status

from agent_service.endpoints.authz_helper import User, parse_header
from agent_service.endpoints.models import (
    DeleteMemoryResponse,
    GetAutocompleteItemsRequest,
    GetAutocompleteItemsResponse,
    GetMemoryContentResponse,
    ListMemoryItemsResponse,
    RenameMemoryRequest,
    RenameMemoryResponse,
)
from agent_service.endpoints.routers.utils import get_agent_svc_impl

router = APIRouter(prefix="/api/memory")


@router.get(
    "/list-memory-items",
    response_model=ListMemoryItemsResponse,
    status_code=status.HTTP_200_OK,
)
async def list_memory_items(user: User = Depends(parse_header)) -> ListMemoryItemsResponse:
    """
    List memory items
    """
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.list_memory_items(user_id=user.user_id)


@router.post(
    "/get-autocomplete-items",
    response_model=GetAutocompleteItemsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_autocomplete_items(
    req: GetAutocompleteItemsRequest, user: User = Depends(parse_header)
) -> GetAutocompleteItemsResponse:
    """
    Gets autocomplete items
    """
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_autocomplete_items(
        user_id=user.user_id, text=req.text, memory_type=req.memory_type
    )


@router.get(
    "/get-memory-content/{type}/{id}",
    response_model=GetMemoryContentResponse,
    status_code=status.HTTP_200_OK,
)
async def get_memory_content(
    type: str, id: str, user: User = Depends(parse_header)
) -> GetMemoryContentResponse:
    """
    Gets preview of memory content (output in text or table form)
    Args:
        type (str): memory type (portfolio / watchlist)
        id (str): the ID of the memory type
    """
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_memory_content(user_id=user.user_id, type=type, id=id)


@router.delete(
    "/delete-memory/{type}/{id}",
    response_model=DeleteMemoryResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_memory(
    type: str, id: str, user: User = Depends(parse_header)
) -> DeleteMemoryResponse:
    """
    Delete memory item
    """
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.delete_memory(user_id=user.user_id, type=type, id=id)


@router.post(
    "/rename-memory",
    response_model=RenameMemoryResponse,
    status_code=status.HTTP_200_OK,
)
async def rename_memory(
    req: RenameMemoryRequest, user: User = Depends(parse_header)
) -> RenameMemoryResponse:
    """
    Rename memory item
    """
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.rename_memory(
        user_id=user.user_id, type=req.type, id=req.id, new_name=req.new_name
    )
