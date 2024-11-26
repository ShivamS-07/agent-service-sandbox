from fastapi import APIRouter, Depends, HTTPException
from starlette import status

from agent_service.endpoints.authz_helper import (
    User,
    parse_header,
    validate_user_agent_access,
)
from agent_service.endpoints.models import (
    GetAgentDebugInfoResponse,
    GetDebugToolArgsResponse,
    GetDebugToolResultResponse,
    GetPlanRunDebugInfoResponse,
    GetToolLibraryResponse,
    ModifyPlanRunArgsRequest,
    ModifyPlanRunArgsResponse,
)
from agent_service.endpoints.routers.utils import get_agent_svc_impl
from agent_service.utils.feature_flags import is_user_agent_admin

router = APIRouter(prefix="/api/debug")


@router.get("/{agent_id}", response_model=GetAgentDebugInfoResponse, status_code=status.HTTP_200_OK)
async def get_agent_debug_info(
    agent_id: str, user: User = Depends(parse_header)
) -> GetAgentDebugInfoResponse:
    agent_svc_impl = get_agent_svc_impl()

    if not user.is_super_admin and not await is_user_agent_admin(
        user.user_id, async_db=agent_svc_impl.pg
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return await agent_svc_impl.get_agent_debug_info(agent_id=agent_id)


@router.get(
    "/get-plan-run-debug-info/{agent_id}/{plan_run_id}",
    response_model=GetPlanRunDebugInfoResponse,
    status_code=status.HTTP_200_OK,
)
async def get_plan_run_debug_info(
    agent_id: str, plan_run_id: str, user: User = Depends(parse_header)
) -> GetPlanRunDebugInfoResponse:
    """Return detailed information about a plan run for debugging purposes,
    include the list of tools, inputs, outputs
    """
    agent_svc_impl = get_agent_svc_impl()

    if not (
        user.is_super_admin or await is_user_agent_admin(user.user_id, async_db=agent_svc_impl.pg)
    ):
        await validate_user_agent_access(user.user_id, agent_id, async_db=agent_svc_impl.pg)

    return await agent_svc_impl.get_plan_run_debug_info(agent_id=agent_id, plan_run_id=plan_run_id)


@router.post(
    "/modify-plan-run-args/{agent_id}/{plan_run_id}",
    response_model=ModifyPlanRunArgsResponse,
    status_code=status.HTTP_200_OK,
)
async def modify_plan_run_args(
    agent_id: str,
    plan_run_id: str,
    req: ModifyPlanRunArgsRequest,
    user: User = Depends(parse_header),
) -> ModifyPlanRunArgsResponse:
    """Duplicate the plan with modified input variables and rerun it"""
    agent_svc_impl = get_agent_svc_impl()

    if not (
        user.is_super_admin or await is_user_agent_admin(user.user_id, async_db=agent_svc_impl.pg)
    ):
        await validate_user_agent_access(user.user_id, agent_id, async_db=agent_svc_impl.pg)

    return await agent_svc_impl.modify_plan_run_args(
        agent_id=agent_id, plan_run_id=plan_run_id, user_id=user.user_id, req=req
    )


@router.get(
    "/args/{replay_id}",
    response_model=GetDebugToolArgsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_debug_args(
    replay_id: str, user: User = Depends(parse_header)
) -> GetDebugToolArgsResponse:
    agent_svc_impl = get_agent_svc_impl()

    if not user.is_super_admin and not await is_user_agent_admin(
        user.user_id, async_db=agent_svc_impl.pg
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return await agent_svc_impl.get_debug_tool_args(replay_id=replay_id)


@router.get(
    "/result/{replay_id}",
    response_model=GetDebugToolResultResponse,
    status_code=status.HTTP_200_OK,
)
async def get_agent_debug_result(
    replay_id: str, user: User = Depends(parse_header)
) -> GetDebugToolResultResponse:
    agent_svc_impl = get_agent_svc_impl()

    if not user.is_super_admin and not await is_user_agent_admin(
        user.user_id, async_db=agent_svc_impl.pg
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized"
        )
    return await agent_svc_impl.get_debug_tool_result(replay_id=replay_id)


@router.get(
    "/tool/get-tool-library",
    response_model=GetToolLibraryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_tool_library(user: User = Depends(parse_header)) -> GetToolLibraryResponse:
    agent_svc_impl = get_agent_svc_impl()

    return await agent_svc_impl.get_tool_library(user=user)
