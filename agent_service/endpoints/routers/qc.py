import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from starlette import status

from agent_service.endpoints.authz_helper import User, parse_header
from agent_service.endpoints.models import (
    AddAgentQCRequest,
    AddAgentQCResponse,
    AgentQC,
    GetAgentsQCRequest,
    GetLiveAgentsQCResponse,
    QueryWithBreakdown,
    SearchAgentQCRequest,
    SearchAgentQCResponse,
    UpdateAgentQCRequest,
    UpdateAgentQCResponse,
)
from agent_service.endpoints.routers.utils import get_agent_svc_impl
from agent_service.utils.feature_flags import user_has_qc_tool_access

router = APIRouter(prefix="/api/agent/qc")


@router.get("/{id}", response_model=List[AgentQC], status_code=status.HTTP_200_OK)
async def get_qc_agent_by_id(id: str, user: User = Depends(parse_header)) -> List[AgentQC]:
    """
    Get QC Agent by ID

    Args:
        id (UUID4): The ID of the agent QC.

    Returns:
        A list of AgentQC objects.
    """
    # Validate user access to QC tool
    agent_svc_impl = get_agent_svc_impl()
    if not await user_has_qc_tool_access(user_id=user.user_id, async_db=agent_svc_impl.pg):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    # Call the function to retrieve agent QC by ID
    agent_qcs = await agent_svc_impl.get_agent_qc_by_ids([id])

    # Return the list of AgentQC objects
    return agent_qcs


@router.get("/user/{user_id}", response_model=List[AgentQC], status_code=status.HTTP_200_OK)
async def get_qc_agent_by_user(user_id: str, user: User = Depends(parse_header)) -> List[AgentQC]:
    """
    Get QC Agents by User ID

    Args:
        user_id (UUID4): The ID of the user whose QC agents are being requested.

    Returns:
        A list of AgentQC objects.
    """
    # Validate user access to QC tool
    agent_svc_impl = get_agent_svc_impl()
    if not await user_has_qc_tool_access(user_id=user.user_id, async_db=agent_svc_impl.pg):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    # Call the function to retrieve agent QCs by user_id
    agent_qcs = await agent_svc_impl.get_agent_qc_by_user_ids([user_id])

    # Return the list of AgentQC objects
    return agent_qcs


@router.post("/search", response_model=SearchAgentQCResponse, status_code=status.HTTP_200_OK)
async def search_agent_qc(
    req: SearchAgentQCRequest, user: User = Depends(parse_header)
) -> SearchAgentQCResponse:
    """
    Search Agent QC records based on various filters

    Args:
        start_date (Optional[date]): The date to filter by.
        end_date (Optional[date]): The date to filter by.
        use_case (Optional[str]): The use case to filter by.
        score_rating (Optional[int]): The score rating to filter by.
        tool_failed (Optional[bool]): Whether the tool failed.
        problem_type (Optional[str]): The type of problem to filter by.

    Returns:
        A list of AgentQC records matching the search criteria.
    """
    # Validate user access to QC tool
    agent_svc_impl = get_agent_svc_impl()
    if not await user_has_qc_tool_access(user_id=user.user_id, async_db=agent_svc_impl.pg):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    # Call the service function to search based on the filters
    agent_qcs, total_agent_qcs = await agent_svc_impl.search_agent_qcs(
        filter_criteria=req.filter_criteria,
        search_criteria=req.search_criteria,
        pagination=req.pagination,
    )

    # Return the list of AgentQC records in the response model format
    return SearchAgentQCResponse(agent_qcs=agent_qcs, total_agent_qcs=total_agent_qcs)


@router.post(
    "/get-live-agents", response_model=GetLiveAgentsQCResponse, status_code=status.HTTP_200_OK
)
async def get_live_agents_qc(
    req: GetAgentsQCRequest, user: User = Depends(parse_header)
) -> GetLiveAgentsQCResponse:
    # Validate user access to QC tool
    agent_svc_impl = get_agent_svc_impl()
    if not await user_has_qc_tool_access(user_id=user.user_id, async_db=agent_svc_impl.pg):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    return await agent_svc_impl.get_live_agents_qc(req=req)


@router.post("/add", response_model=AddAgentQCResponse, status_code=status.HTTP_200_OK)
async def add_agent_qc(
    req: AddAgentQCRequest, user: User = Depends(parse_header)
) -> AddAgentQCResponse:
    # Validate user access to QC tool
    agent_svc_impl = get_agent_svc_impl()
    if not await user_has_qc_tool_access(user_id=user.user_id, async_db=agent_svc_impl.pg):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    agent_id = req.agent_id
    user_id = req.user_id
    # Call the service function to add agent_qc, returns success status and the inserted agent_qc
    success, agent_qc = await agent_svc_impl.add_qc_agent(agent_id, user_id)

    return AddAgentQCResponse(success=success, agent_qc=agent_qc)


@router.post("/update", response_model=UpdateAgentQCResponse, status_code=status.HTTP_200_OK)
async def update_agent_qc(
    req: UpdateAgentQCRequest, user: User = Depends(parse_header)
) -> UpdateAgentQCResponse:
    # Validate user access to QC tool
    agent_svc_impl = get_agent_svc_impl()
    if not await user_has_qc_tool_access(user_id=user.user_id, async_db=agent_svc_impl.pg):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )

    agent_qc = req.agent_qc
    # Call the service function to search based on the filters
    res = await agent_svc_impl.update_qc_agent(agent_qc)

    # Return the list of AgentQC records in the response model format
    return UpdateAgentQCResponse(success=res)


@router.get(
    "/stats/query-breakdown-deep-dive/{date}",
    response_model=List[QueryWithBreakdown],
    status_code=status.HTTP_200_OK,
)
async def get_query_breakdown_deep_dive(
    date: datetime.datetime,
    user: User = Depends(parse_header),
) -> List[QueryWithBreakdown]:
    # Validate user access to QC tool
    agent_svc_impl = get_agent_svc_impl()
    if not await user_has_qc_tool_access(user_id=user.user_id, async_db=agent_svc_impl.pg):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized to use QC tool"
        )
    res = await agent_svc_impl.pg.get_all_usecase_rating_within_week(date)
    return res
