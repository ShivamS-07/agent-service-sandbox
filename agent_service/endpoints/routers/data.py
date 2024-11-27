from fastapi import APIRouter, Depends, HTTPException
from starlette import status

from agent_service.endpoints.authz_helper import User, parse_header
from agent_service.endpoints.models import (
    ExperimentalGetFormulaDataRequest,
    ExperimentalGetFormulaDataResponse,
    GetAvailableVariablesResponse,
    GetVariableCoverageRequest,
    GetVariableCoverageResponse,
    GetVariableHierarchyResponse,
)
from agent_service.endpoints.routers.utils import get_agent_svc_impl
from agent_service.utils.feature_flags import user_has_variable_dashboard_access

router = APIRouter(prefix="/api/data")


# variables/data endpoints
@router.get(
    "/variables/available-variables",
    response_model=GetAvailableVariablesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_available_variable(
    user: User = Depends(parse_header),
) -> GetAvailableVariablesResponse:
    """
    Retrieves relevant metadata about all variables available to the user.
    """
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_all_available_variables(user=user)


@router.get(
    "/variables/hierarchy",
    response_model=GetVariableHierarchyResponse,
    status_code=status.HTTP_200_OK,
)
async def get_all_variable_hierarchy(
    user: User = Depends(parse_header),
) -> GetVariableHierarchyResponse:
    """
    Retrieves all variable display hierarchies in a flat format.
    """
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_variable_hierarchy(user=user)


@router.post(
    "/variables/coverage",
    response_model=GetVariableCoverageResponse,
    status_code=status.HTTP_200_OK,
)
async def get_variable_coverage(
    req: GetVariableCoverageRequest, user: User = Depends(parse_header)
) -> GetVariableCoverageResponse:
    """
    Retrieves coverage information for all available variables.
    Default universe SPY
    """
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.get_variable_coverage(
        user=user, feature_ids=req.feature_ids, universe_id=req.universe_id
    )


@router.post(
    "/variables/evaluate-formula",
    response_model=ExperimentalGetFormulaDataResponse,
    status_code=status.HTTP_200_OK,
)
async def experimental_variable_evaluate_formula(
    req: ExperimentalGetFormulaDataRequest, user: User = Depends(parse_header)
) -> ExperimentalGetFormulaDataResponse:
    """
    Gets a formatted output for an experimental variable formula mode.
    """
    agent_svc_impl = get_agent_svc_impl()
    has_access = await user_has_variable_dashboard_access(
        user_id=user.user_id, async_db=agent_svc_impl.pg
    )
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is not authorized."
        )

    return await agent_svc_impl.experimental_get_formula_data_impl(
        user=user,
        markdown_formula=req.markdown_formula,
        stock_ids=req.gbi_ids,
        from_date=req.from_date,
        to_date=req.to_date,
    )
