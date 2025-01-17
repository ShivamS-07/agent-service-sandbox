import argparse
import asyncio
import datetime
import logging
import sys
from pprint import pprint
from typing import Optional
from uuid import uuid4

from gbi_common_py_utils.utils.environment import get_environment_tag

# from agent_service.agent_service_impl import AgentServiceImpl
from agent_service.endpoints.models import AgentInfo
from agent_service.io_type_utils import ComplexIOBase, IOType
from agent_service.planner.executor import run_execution_plan_local
from agent_service.planner.plan_creation import (
    create_execution_plan_local,
    update_execution_after_input,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.agent_event_utils import publish_agent_name, send_chat_message
from agent_service.utils.agent_name import generate_name_for_agent
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.date_utils import enable_mock_time, get_now_utc, set_mock_time
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.output_utils.utils import output_for_log
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, get_psql

logger = logging.getLogger(__name__)

env = get_environment_tag()
base_url = "alfa.boosted.ai" if env == "ALPHA" else "agent-dev.boosted.ai"
channel = "alfa-client-queries" if env == "ALPHA" else "alfa-client-queries-dev"


async def soft_delete_agent(agent_id: str) -> None:
    # agent_service_impl.py::delete_agent()
    async_pg = AsyncPostgresBase()
    async_db = AsyncDB(async_pg)

    await async_db.delete_agent_by_id(agent_id)  # soft delete
    await send_chat_message(
        message=Message(
            agent_id=agent_id,
            message="Analyst has been deleted successfully.",
            is_user_message=False,
            visible_to_llm=False,
        ),
        db=async_db,
    )


async def generate_agent_name_and_store(user_id: str, agent_id: str, user_msg: Message) -> str:
    # agent_service_impl.py::_generate_agent_name_and_store()
    async_pg = AsyncPostgresBase()
    pg = AsyncDB(async_pg)

    logger.info("Getting existing agents' names")
    existing_agents = await pg.get_existing_agents_names(user_id)

    logger.info("Calling GPT to generate agent name")
    name = await generate_name_for_agent(
        agent_id=agent_id,
        chat_context=ChatContext(messages=[user_msg]),
        existing_names=existing_agents,
        user_id=user_id,
    )

    logger.info(f"Updating agent name to {name} in DB")

    await asyncio.gather(
        pg.update_agent_name(agent_id=agent_id, agent_name=name),
        publish_agent_name(agent_id=agent_id, agent_name=name),
    )
    return name


def remove_citations_from_output_iotype(output: IOType) -> None:
    # this is just to make the output more readable
    if isinstance(output, list):
        for suboutput in output:
            remove_citations_from_output_iotype(suboutput)
    elif isinstance(output, ComplexIOBase):
        for history_entry in output.history:
            history_entry.citations = []
        if hasattr(output, "val"):
            remove_citations_from_output_iotype(output.val)


async def gen_and_run_plan(
    user_id: str,
    agent_id: Optional[str],
    prompt: Optional[str] = None,
    run_plan_without_confirmation: bool = True,
    verbose: bool = False,
    do_chat: bool = False,
    replan_execution_error: bool = False,
    multiple_inputs: bool = False,
    use_sample_plans: bool = True,
    mock_automation: bool = False,
    exclude_citations: bool = False,
    as_of_date: Optional[datetime.date] = None,
) -> None:
    if not prompt:
        prompt = input("Enter a prompt for the agent> ")

    print(f"Creating execution plan for prompt: '{prompt}'")

    if as_of_date:
        start_time = datetime.datetime.combine(as_of_date, datetime.time(hour=7, minute=35))
        enable_mock_time()
        set_mock_time(start_time)
        print(f"overriding current time to: {start_time=}")

    now = get_now_utc()
    message = Message(message=prompt, is_user_message=True)
    chat = ChatContext(messages=[message])

    plan_id = str(uuid4())

    db = get_psql(skip_commit=False)
    if not agent_id:
        agent_id = str(uuid4())
        create_agent(user_id, agent_id)

    user_msg = Message(
        agent_id=agent_id,
        message=prompt,
        is_user_message=True,
        message_time=now,
    )

    await generate_agent_name_and_store(user_id=user_id, agent_id=agent_id, user_msg=user_msg)

    db.insert_chat_messages([user_msg])

    plan_run_id = str(uuid4())
    plan = await create_execution_plan_local(
        agent_id=agent_id,
        plan_id=plan_id,
        user_id=user_id,
        skip_db_commit=False,
        skip_task_cache=False,
        run_plan_in_prefect_immediately=False,
        run_tasks_without_prefect=True,
        chat_context=chat,
        do_chat=do_chat,
        use_sample_plans=use_sample_plans,
        plan_run_id=plan_run_id,
    )
    if plan is None:
        print("failed to create plan")
        return
    print(f"created {plan_id=}, {agent_id=}")
    if not run_plan_without_confirmation:
        cont = input("Shall I continue? (y/n)> ")
        if cont.lower() != "y":
            print(f"Exiting, and cleaning up {agent_id=}")
            await soft_delete_agent(agent_id)
            return

    print(f"Running execution plan... {plan_id=}, {plan_run_id=}")
    context = PlanRunContext(
        agent_id=agent_id,
        plan_id=plan_id,
        user_id=user_id,
        plan_run_id=plan_run_id,
        skip_db_commit=False,
        skip_task_cache=False,
        run_tasks_without_prefect=True,
    )

    if as_of_date:
        context.as_of_date = start_time

    context.chat = chat
    print(f"{context=}")
    output, _ = await run_execution_plan_local(
        plan=plan,
        context=context,
        do_chat=do_chat,
        log_all_outputs=verbose,
        replan_execution_error=replan_execution_error,
        scheduled_by_automation=mock_automation,
    )
    print(f"Successfully ran execution plan: {agent_id=} {plan_id=} {plan_run_id=}")
    if exclude_citations:
        remove_citations_from_output_iotype(output)
    print("Output from main run:")
    print(output_for_log(output))

    print(f"created: https://{base_url}/chat/{agent_id}")
    # untested
    if multiple_inputs:
        while True:
            print(f"Successfully ran execution plan: {agent_id=} {plan_id=} {plan_run_id=}")
            prompt = input("Enter a follow up prompt for the agent (type n to stop)> ")

            if prompt == "n":
                break
            print(f"Updating execution plan given prompt: '{prompt}'")

            now = get_now_utc()

            new_user_msg = Message(
                agent_id=agent_id,
                message=prompt,
                is_user_message=True,
                message_time=now,
            )

            db.insert_chat_messages([new_user_msg])
            replanning_output = await update_execution_after_input(
                agent_id,
                user_id,
                skip_db_commit=False,
                skip_task_cache=False,
                run_plan_in_prefect_immediately=False,
                run_tasks_without_prefect=True,
                do_chat=do_chat,
                use_sample_plans=use_sample_plans,
            )
            if replanning_output is None:
                print("failed to create plan or no replanning needed")
                continue
            new_plan_id, plan, action = replanning_output
            print(action)
            if not run_plan_without_confirmation:
                cont = input("Shall I continue? (y/n)> ")
                if cont.lower() != "y":
                    print("Exiting...")
                    return

            plan_id = new_plan_id
            new_plan_run_id = str(uuid4())
            plan_run_id = new_plan_run_id
            context = PlanRunContext(
                agent_id=agent_id,
                plan_id=new_plan_id,
                user_id=user_id,
                plan_run_id=new_plan_id,
                skip_db_commit=False,
                skip_task_cache=False,
                run_tasks_without_prefect=True,
                chat=db.get_chats_history_for_agent(agent_id),
            )

            output, _ = await run_execution_plan_local(
                plan=plan,
                context=context,
                do_chat=do_chat,
                log_all_outputs=verbose,
                replan_execution_error=replan_execution_error,
            )
            print("Output from follow up run:")
            pprint(output)

    if do_chat:
        print("Chat:")
        print(db.get_chats_history_for_agent(agent_id).get_gpt_input())


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
    parser.add_argument("-d", "--as-of-date", type=str, default="")
    parser.add_argument("-u", "--user_id", type=str)  # useful if tool needs user
    parser.add_argument("-c", "--do_chat", action="store_true", default=False)
    parser.add_argument("-e", "--retry_on_execution_error", action="store_true", default=False)
    parser.add_argument("-i", "--allow_additional_input", action="store_true", default=False)
    parser.add_argument("-n", "--not_use_sample_plans", action="store_true", default=False)
    parser.add_argument("-x", "--exclude_citations", action="store_true", default=False)
    parser.add_argument(
        "-a",
        "--mock_automation",
        action="store_true",
        default=False,
        help="If true, pretend that the plan was kicked off via automation",
    )
    return parser.parse_args()


def create_agent(user_id: str, agent_id: str) -> None:
    agent = AgentInfo(
        agent_id=agent_id,
        user_id=user_id,
        agent_name=DEFAULT_AGENT_NAME,
        created_at=get_now_utc(),
        last_updated=get_now_utc(),
        deleted=False,
    )

    db = get_psql(skip_commit=False)
    db.insert_agent(agent)
    print(f"created {agent_id=}, {agent=}")


async def main() -> None:
    args = parse_args()
    continue_agent = True

    as_of_date = args.as_of_date
    if as_of_date:
        try:
            as_of_date = datetime.date.fromisoformat(as_of_date)
        except Exception:
            as_of_date = None
            pass
    else:
        as_of_date = None

    agent_id = str(uuid4())

    if as_of_date:
        # teleport to the past before creating the agent in DB
        start_time = datetime.datetime.combine(as_of_date, datetime.time(hour=7, minute=35))
        enable_mock_time()
        set_mock_time(start_time)
        print(f"overriding current time to: {start_time=}")

    create_agent(user_id=args.user_id, agent_id=agent_id)
    while continue_agent:
        # add agent_id to the input
        await gen_and_run_plan(
            user_id=args.user_id,
            agent_id=agent_id,
            prompt=args.prompt,
            run_plan_without_confirmation=args.run_plan_no_confirm,
            verbose=args.verbose,
            do_chat=True,
            replan_execution_error=args.retry_on_execution_error,
            multiple_inputs=args.allow_additional_input,
            use_sample_plans=not args.not_use_sample_plans,
            mock_automation=args.mock_automation,
            exclude_citations=args.exclude_citations,
            as_of_date=as_of_date,
        )
        while True:
            should_continue = input("Try another prompt? (y/n)> ")
            if should_continue.lower() == "y":
                break
            elif should_continue.lower() == "n":
                continue_agent = False
                break


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    init_stdout_logging()
    asyncio.run(main())
