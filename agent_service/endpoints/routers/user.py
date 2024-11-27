from fastapi import APIRouter, Depends, Request
from starlette import status

from agent_service.endpoints.authz_helper import User, parse_header
from agent_service.endpoints.models import (
    AgentUserSettingsSetRequest,
    GetAccountInfoResponse,
    GetTeamAccountsResponse,
    GetUsersRequest,
    GetUsersResponse,
    UpdateUserRequest,
    UpdateUserResponse,
    UserHasAccessResponse,
)
from agent_service.endpoints.routers.utils import get_agent_svc_impl
from agent_service.utils.async_utils import run_async_background
from agent_service.utils.email_utils import AgentEmail
from agent_service.utils.feature_flags import is_user_agent_admin
from agent_service.utils.user_metadata import is_user_first_login

router = APIRouter(prefix="/api/user")


# Account Endpoints
@router.patch("/settings", response_model=UpdateUserResponse, status_code=status.HTTP_200_OK)
async def update_user_settings(
    req: AgentUserSettingsSetRequest, user: User = Depends(parse_header)
) -> UpdateUserResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.update_user_settings(user=user, req=req)


@router.post("/update-user", response_model=UpdateUserResponse, status_code=status.HTTP_200_OK)
async def update_user(
    req: UpdateUserRequest, user: User = Depends(parse_header)
) -> UpdateUserResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.update_user(
        user_id=user.user_id, name=req.name, username=req.username, email=req.email
    )


@router.get(
    "/get-account-info", response_model=GetAccountInfoResponse, status_code=status.HTTP_200_OK
)
async def get_account_info(user: User = Depends(parse_header)) -> GetAccountInfoResponse:
    agent_svc_impl = get_agent_svc_impl()
    account = await agent_svc_impl.get_account_info(user=user)
    return GetAccountInfoResponse(account=account)


@router.post("/get-users", response_model=GetUsersResponse, status_code=status.HTTP_200_OK)
async def get_users_info(
    req: GetUsersRequest, user: User = Depends(parse_header)
) -> GetUsersResponse:
    agent_svc_impl = get_agent_svc_impl()
    accounts = await agent_svc_impl.get_users_info(user=user, user_ids=req.user_ids)
    return GetUsersResponse(accounts=accounts)


@router.get(
    "/get-team-accounts", response_model=GetTeamAccountsResponse, status_code=status.HTTP_200_OK
)
async def get_team_accounts(user: User = Depends(parse_header)) -> GetTeamAccountsResponse:
    agent_svc_impl = get_agent_svc_impl()
    accounts = await agent_svc_impl.get_valid_notification_users(user_id=user.user_id)
    return GetTeamAccountsResponse(accounts=accounts)


@router.get("/has-access", response_model=UserHasAccessResponse, status_code=status.HTTP_200_OK)
async def get_user_has_access(
    request: Request, user: User = Depends(parse_header)
) -> UserHasAccessResponse:
    agent_svc_impl = get_agent_svc_impl()
    is_admin = (
        user.is_admin
        or user.is_super_admin
        or (await is_user_agent_admin(user.user_id, async_db=agent_svc_impl.pg))
    )

    if not is_admin:
        has_access = await agent_svc_impl.get_user_has_alfa_access(user=user)
    else:
        has_access = True

    if not has_access:
        return UserHasAccessResponse(success=False)

    # make sure user is not spoofed and it is their first login
    if request.headers.get("realuserid") == user.user_id and await is_user_first_login(
        user_id=user.user_id
    ):
        run_async_background(
            AgentEmail(db=agent_svc_impl.pg).send_welcome_email(user_id=user.user_id)
        )

    return UserHasAccessResponse(success=True)
