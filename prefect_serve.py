# mypy: ignore-errors
from prefect import serve

from agent_service.planner.constants import (
    CREATE_EXECUTION_PLAN_FLOW_NAME,
    RUN_EXECUTION_PLAN_FLOW_NAME,
)
from agent_service.planner.executor import create_execution_plan, run_execution_plan

if __name__ == "__main__":
    create_execution_plan_deploy = create_execution_plan.to_deployment(
        name=CREATE_EXECUTION_PLAN_FLOW_NAME
    )
    run_execution_plan_deploy = run_execution_plan.to_deployment(name=RUN_EXECUTION_PLAN_FLOW_NAME)
    serve(create_execution_plan_deploy, run_execution_plan_deploy)
