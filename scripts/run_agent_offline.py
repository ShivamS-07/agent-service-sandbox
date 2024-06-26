import argparse
import asyncio
from pprint import pprint
from typing import Optional
from uuid import uuid4

from agent_service.endpoints.models import AgentMetadata
from agent_service.planner.executor import (
    create_execution_plan_local,
    run_execution_plan_local,
    update_execution_after_input,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, get_psql


async def gen_and_run_plan(
    prompt: Optional[str] = None,
    run_plan_without_confirmation: bool = True,
    verbose: bool = False,
    user_id: Optional[str] = None,
    do_chat: bool = False,
    replan_execution_error: bool = False,
    multiple_inputs: bool = False,
    use_sample_plans: bool = True,
    mock_automation: bool = False,
) -> None:
    if not prompt:
        prompt = input("Enter a prompt for the agent> ")

    print(f"Creating execution plan for prompt: '{prompt}'")

    now = get_now_utc()
    chat = ChatContext(messages=[Message(message=prompt, is_user_message=True)])
    agent_id = str(uuid4())
    if user_id is None:
        user_id = str(uuid4())
    plan_id = str(uuid4())
    agent = AgentMetadata(
        agent_id=agent_id,
        user_id=user_id,
        agent_name=DEFAULT_AGENT_NAME,
        created_at=get_now_utc(),
        last_updated=get_now_utc(),
    )
    user_msg = Message(
        agent_id=agent.agent_id,
        message=prompt,
        is_user_message=True,
        message_time=now,
    )

    db = get_psql(skip_commit=True)
    db.insert_agent(agent)
    db.insert_chat_messages([user_msg])

    plan = await create_execution_plan_local(
        agent_id=agent_id,
        plan_id=plan_id,
        user_id=user_id,
        skip_db_commit=True,
        skip_task_cache=True,
        run_plan_in_prefect_immediately=False,
        run_tasks_without_prefect=True,
        chat_context=chat,
        do_chat=do_chat,
        use_sample_plans=use_sample_plans,
    )
    if plan is None:
        print("failed to create plan")
        return
    if not run_plan_without_confirmation:
        cont = input("Shall I continue? (y/n)> ")
        if cont.lower() != "y":
            print("Exiting...")
            return

    print("Running execution plan...")
    context = PlanRunContext(
        agent_id=agent_id,
        plan_id=plan_id,
        user_id=user_id,
        plan_run_id=str(uuid4()),
        skip_db_commit=True,
        skip_task_cache=True,
        run_tasks_without_prefect=True,
    )
    context.chat = chat
    output = await run_execution_plan_local(
        plan=plan,
        context=context,
        do_chat=do_chat,
        log_all_outputs=verbose,
        replan_execution_error=replan_execution_error,
        scheduled_by_automation=mock_automation,
    )
    print("Output from main run:")
    pprint(output)

    if multiple_inputs:
        while True:
            prompt = input("Enter a follow up prompt for the agent (type n to stop)> ")

            if prompt == "n":
                break
            print(f"Updating execution plan given prompt: '{prompt}'")

            now = get_now_utc()

            new_user_msg = Message(
                agent_id=agent.agent_id,
                message=prompt,
                is_user_message=True,
                message_time=now,
            )

            db.insert_chat_messages([new_user_msg])
            replanning_output = await update_execution_after_input(
                agent_id,
                user_id,
                skip_db_commit=True,
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

            context = PlanRunContext(
                agent_id=agent_id,
                plan_id=new_plan_id,
                user_id=user_id,
                plan_run_id=str(uuid4()),
                skip_db_commit=True,
                skip_task_cache=True,
                run_tasks_without_prefect=True,
                chat=db.get_chats_history_for_agent(agent_id),
            )

            output = await run_execution_plan_local(
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
    parser.add_argument("-u", "--user_id", type=str)  # useful if tool needs user
    parser.add_argument("-c", "--do_chat", action="store_true", default=False)
    parser.add_argument("-e", "--retry_on_execution_error", action="store_true", default=False)
    parser.add_argument("-i", "--allow_additional_input", action="store_true", default=False)
    parser.add_argument("-n", "--not_use_sample_plans", action="store_true", default=False)
    parser.add_argument(
        "-a",
        "--mock_automation",
        action="store_true",
        default=False,
        help="If true, pretend that the plan was kicked off via automation",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    continue_agent = True
    while continue_agent:
        await gen_and_run_plan(
            prompt=args.prompt,
            run_plan_without_confirmation=args.run_plan_no_confirm,
            verbose=args.verbose,
            user_id=args.user_id,
            do_chat=args.do_chat,
            replan_execution_error=args.retry_on_execution_error,
            multiple_inputs=args.allow_additional_input,
            use_sample_plans=not args.not_use_sample_plans,
            mock_automation=args.mock_automation,
        )
        while True:
            should_continue = input("Try another prompt? (y/n)> ")
            if should_continue.lower() == "y":
                break
            elif should_continue.lower() == "n":
                continue_agent = False
                break


if __name__ == "__main__":
    init_stdout_logging()
    asyncio.run(main())
