import unittest
from typing import Optional, Tuple

from agent_service.GPT.requests import set_use_global_stub
from agent_service.io_type_utils import IOType
from agent_service.planner.executor import run_execution_plan_local
from agent_service.planner.plan_creation import create_execution_plan_local
from agent_service.types import ChatContext, Message, PlanRunContext


async def gen_and_run_plan(prompt: str, verbose: bool = False) -> Tuple[str, Optional[IOType]]:
    if not prompt:
        prompt = input("Enter a prompt for the agent> ")

    # print(f"Creating execution plan for prompt: '{prompt}'")
    chat = ChatContext(messages=[Message(message=prompt, is_user_message=True)])
    try:
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
    except Exception:
        print("unable to formulate plan!")
        raise
    # print("Created a plan with the following steps:")
    # print("-----")
    # print(plan.get_formatted_plan())
    # print("-----")
    # print("Running execution plan...")
    context = PlanRunContext.get_dummy()
    context.chat = chat
    try:
        output, _ = await run_execution_plan_local(
            plan=plan, context=context, do_chat=False, log_all_outputs=verbose
        )
    except Exception:
        print("problem running plan:", plan.get_formatted_plan())
        raise

    # print("Got output:")
    # pprint(output)
    plan_str = plan.get_formatted_plan()
    return plan_str, output


class SimpleQueryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # uncomment for easier debugging
        # from agent_service.utils.logs import init_stdout_logging
        # init_stdout_logging()
        set_use_global_stub(False)
        self.context = PlanRunContext.get_dummy()

    @unittest.skip(
        "these tests usually pass but sometimes fail, skip until planner becomes more robust"
    )
    async def test_check_no_throw(self):
        qries = [
            "what is ibm's price",
            "get the financial stocks in sp500",
            "write commentary on inflation",
        ]

        for q in qries:
            with self.subTest(q=q):
                try:
                    await gen_and_run_plan(prompt=q, verbose=True)
                    print(f"TEST PASSED: for qry: {q}")
                except Exception as e:
                    print(f"TEST FAILED: {repr(e)} for qry: {q}")
                    raise

    @unittest.skip(
        """these should fail and often do, but the planner creates unique plans
 for the same input  very often,
 they are wrong plans but they don't always throw..."""
    )
    async def test_check_correctly_throws(self):
        qries = [
            "what is askjfhaslkdjfhsjkh's price",
            "what is IBM's askjfhaslkdjfhsjkh statistic",
        ]

        for q in qries:
            with self.subTest(q=q):
                try:
                    plan_str, output = await gen_and_run_plan(prompt=q, verbose=True)
                except Exception as e:
                    print(f"TEST PASSED: caught an expected exception {repr(e)} for qry: {q}")
                    continue

                self.assertTrue(
                    False,
                    f"TEST FAILED: exception was expected but got none for qry: {q},\n"
                    f"plan: {plan_str},\n output:{str(output)[:200]}",
                )
