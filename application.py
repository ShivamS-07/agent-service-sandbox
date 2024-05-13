import argparse
import logging

import uvicorn
from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from agent_service.endpoints.authz_helper import (
    User,
    get_keyid_to_key_map,
    parse_header,
    validate_user_agent_access,
)
from agent_service.endpoints.models import (
    CreateAgentRequest,
    CreateAgentResponse,
    DeleteAgentRequest,
    DeleteAgentResponse,
    GetAllAgentsResponse,
)
from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import get_psql
from agent_service.utils.sentry_utils import init_sentry

DEFAULT_IP = "0.0.0.0"
DEFAULT_DAL_PORT = 8000
SERVICE_NAME = "AgentService"

logger = logging.getLogger(__name__)

application = FastAPI(title="Agent Service")
application.add_middleware(
    CORSMiddleware,  # Add CORS middleware
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


####################################################################################################
# Test endpoints
####################################################################################################
@application.get("/", response_class=HTMLResponse, status_code=200)
def confirm_working() -> str:
    return "<html>Agent Service is online</html>"


@application.get("/health")
def health() -> str:
    return "OK"


####################################################################################################
# Agent endpoints
####################################################################################################
@application.post(
    "/agent/create-agent", response_model=CreateAgentResponse, status_code=status.HTTP_201_CREATED
)
def create_agent(
    req: CreateAgentRequest, user: User = Depends(parse_header)
) -> CreateAgentResponse:
    agent_id = get_psql().create_agent_for_user(user.user_id)

    # TODO: placeholder
    # 1. connect to GPT to get response
    # 2. store user's prompt and gpt response in db

    return CreateAgentResponse(agent_id=agent_id)


@application.delete(
    "/agent/delete-agent", response_model=DeleteAgentResponse, status_code=status.HTTP_200_OK
)
def delete_agent(
    req: DeleteAgentRequest, user: User = Depends(parse_header)
) -> DeleteAgentResponse:
    validate_user_agent_access(user.user_id, req.agent_id)

    get_psql().delete_agent_by_id(req.agent_id)
    return DeleteAgentResponse(success=True)


@application.get(
    "/agent/get-all-agents", response_model=GetAllAgentsResponse, status_code=status.HTTP_200_OK
)
def get_all_agents(user: User = Depends(parse_header)) -> GetAllAgentsResponse:
    return GetAllAgentsResponse(agents=get_psql().get_user_all_agents(user.user_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--port",
        default=8000,
        type=int,
        required=False,
        help="Port to run the Agent Service on.",
    )
    parser.add_argument(
        "-a",
        "--address",
        type=str,
        required=False,
        default="0.0.0.0",
        help="Address to bind the server to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    init_stdout_logging()
    init_sentry(disable_sentry=not EnvironmentUtils.is_deployed)

    logger.info("Warming up DB connection and JWT key map...")
    get_psql()
    get_keyid_to_key_map()

    logger.info("Starting server...")
    uvicorn.run(application, host=args.address, port=args.port)
