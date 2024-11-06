####################################################################################################
# All endpoints in this file require NO authentication, be careful with these!
####################################################################################################

import logging

from fastapi import APIRouter, FastAPI, status

from agent_service.endpoints.authz_helper import validate_user_plan_run_access
from agent_service.endpoints.models import (
    GetPlanRunOutputResponse,
    RemoveNotificationEmailsRequest,
    RemoveNotificationEmailsResponse,
)

application: FastAPI = None  # type: ignore
router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)


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
    agent_id = await validate_user_plan_run_access(
        request_user_id=None,
        plan_run_id=plan_run_id,
        async_db=application.state.agent_service_impl.pg,
        require_owner=False,
    )
    return await application.state.agent_service_impl.get_plan_run_output(
        agent_id=agent_id, plan_run_id=plan_run_id
    )


@router.post(
    "/agent/notification-emails/remove",
    response_model=RemoveNotificationEmailsResponse,
    status_code=status.HTTP_200_OK,
)
async def remove_agent_notification_emails(
    req: RemoveNotificationEmailsRequest,
) -> RemoveNotificationEmailsResponse:
    agent_id = req.agent_id
    email = req.email
    try:
        await application.state.agent_service_impl.delete_agent_notification_emails(
            agent_id=agent_id, email=email
        )
        return RemoveNotificationEmailsResponse(success=True)
    except Exception:
        logger.exception(
            f"error in removing emails:{req.email} from agent:{req.agent_id} notification"
        )
        return RemoveNotificationEmailsResponse(success=False)
