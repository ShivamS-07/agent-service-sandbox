from fastapi import APIRouter, Depends
from starlette import status

from agent_service.endpoints.authz_helper import User, parse_header, validate_user_agent_access
from agent_service.endpoints.models import (
    SuccessResponse,
    TransformTableOutputRequest,
    TransformTableOutputResponse,
    UpdateTransformationSettingsRequest,
)
from agent_service.endpoints.routers.utils import get_agent_svc_impl

router = APIRouter(prefix="/api/transformation")


@router.post(
    "/transform-table-output",
    response_model=TransformTableOutputResponse,
    status_code=status.HTTP_200_OK,
)
async def transform_table_output(
    req: TransformTableOutputRequest, user: User = Depends(parse_header)
) -> TransformTableOutputResponse:
    agent_svc_impl = get_agent_svc_impl()
    await validate_user_agent_access(user.user_id, req.agent_id, async_db=agent_svc_impl.pg)
    return await agent_svc_impl.transform_table_output(req)


@router.delete(
    "/delete/{agent_id}/{transformation_id}",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_transformation(
    agent_id: str, transformation_id: str, user: User = Depends(parse_header)
) -> SuccessResponse:
    agent_svc_impl = get_agent_svc_impl()
    await validate_user_agent_access(user.user_id, agent_id, async_db=agent_svc_impl.pg)
    return await agent_svc_impl.delete_transformation(transformation_id)


@router.post(
    "/update-transformation-settings",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
)
async def update_transformation_settings(
    req: UpdateTransformationSettingsRequest, user: User = Depends(parse_header)
) -> SuccessResponse:
    agent_svc_impl = get_agent_svc_impl()
    await validate_user_agent_access(user.user_id, req.agent_id, async_db=agent_svc_impl.pg)
    return await agent_svc_impl.update_transformation_settings(req)
