import argparse
import asyncio
from pprint import pprint
from typing import Optional

from agent_service.planner.executor import (
    create_execution_plan_local,
    run_execution_plan_local,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.logs import init_stdout_logging


async def gen_and_run_plan(
    prompt: Optional[str] = None, run_plan_without_confirmation: bool = True, verbose: bool = False
) -> None:
    if not prompt:
        prompt = input("Enter a prompt for the agent> ")

    print(f"Creating execution plan for prompt: '{prompt}'")
    chat = ChatContext(messages=[Message(message=prompt, is_user_message=True)])
    plan = await create_execution_plan_local(
        agent_id="",
        plan_id="",
        user_id="",
        skip_db_commit=True,
        skip_task_cache=True,
        run_plan_in_prefect_immediately=False,
        run_tasks_without_prefect=True,
        chat_context=chat,
    )
    print("Created a plan with the following steps:")
    print("-----")
    print(plan.get_formatted_plan())
    print("-----")
    if not run_plan_without_confirmation:
        cont = input("Shall I continue? (y/n)> ")
        if cont.lower() != "y":
            print("Exiting...")
            return

    print("Running execution plan...")
    context = PlanRunContext.get_dummy()
    context.chat = chat
    output = await run_execution_plan_local(
        plan=plan, context=context, send_chat_when_finished=False, log_all_outputs=verbose
    )
    print("Got output:")
    pprint(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str)
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="If true, log ALL tool outputs."
    )
    parser.add_argument(
        "-y",
        "--run-plan-no-confirm",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    await gen_and_run_plan(
        prompt=args.prompt,
        run_plan_without_confirmation=args.run_plan_no_confirm,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    init_stdout_logging()
    asyncio.run(main())
