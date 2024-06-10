import asyncio

from prefect import serve

from agent_service.planner.constants import (
    CREATE_EXECUTION_PLAN_FLOW_NAME,
    RUN_EXECUTION_PLAN_FLOW_NAME,
)
from agent_service.planner.executor import create_execution_plan, run_execution_plan
from agent_service.utils.logs import init_stdout_logging


async def run() -> None:
    create_execution_plan_deploy = await create_execution_plan.to_deployment(
        name=CREATE_EXECUTION_PLAN_FLOW_NAME
    )
    run_execution_plan_deploy = await run_execution_plan.to_deployment(
        name=RUN_EXECUTION_PLAN_FLOW_NAME
    )
    await serve(create_execution_plan_deploy, run_execution_plan_deploy, query_seconds=0.5)


if __name__ == "__main__":
    init_stdout_logging()
    asyncio.run(run())
