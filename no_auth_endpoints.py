####################################################################################################
# All endpoints in this file require NO authentication, be careful with these!
####################################################################################################

from fastapi import APIRouter, FastAPI, status

from agent_service.endpoints.authz_helper import validate_user_plan_run_access
from agent_service.endpoints.models import GetPlanRunOutputResponse

application: FastAPI = None  # type: ignore
router = APIRouter(prefix="/api")


def initialize_unauthed_endpoints(application_instance: FastAPI) -> None:
    global application
    application = application_instance
    application.include_router(router)


@router.get(
    "/agent/get-plan-run-output/{plan_run_id}",
    response_model=GetPlanRunOutputResponse,
    status_code=status.HTTP_200_OK,
)
async def get_plan_run_output(plan_run_id: str) -> GetPlanRunOutputResponse:
    validate_user_plan_run_access(
        request_user_id=None, plan_run_id=plan_run_id, require_owner=False
    )

    return await application.state.agent_service_impl.get_plan_run_output(plan_run_id=plan_run_id)
