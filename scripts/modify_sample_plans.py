import asyncio
from uuid import uuid4

from agent_service.planner.planner_types import SamplePlan
from agent_service.planner.utils import get_similar_sample_plans
from agent_service.utils.postgres import get_psql

MAIN_PROMPT = "Choose an action (add/list/search/show/delete/help/exit)> "

HELP = """
The add command will add a new input/sample pair
The list command will will list all current inputs, with their ids
The search command will search the current sample plans for matching plans using
the same LLM-based method as the planner does, and output the list of the most
relevant sample inputs
The show command will print the full plan for a particular plan id
The delete command will delete a particular sample plan
The help command shows this help screen
The exit command will exit this script
"""

ADD_PROMPT = "Enter the sample input> "
ADD_FOLLOWUP = "Enter Step {n} of the plan (enter nothing to stop) > "
ADD_CONFIRM = "Are you sure you want to add this sample plan:\n{plan}\n(y/n)> "


DELETE_PROMPT = "Enter the sample plan id you want to delete> "
DELETE_CONFIRM = "Are you sure you want to delete this sample plan:\n{plan}\n(y/n)> "

SHOW_PROMPT = "Enter the sample plan id of the plan you want to inspect> "

SEARCH_PROMPT = "Enter an input that you want to find similar inputs to> "


def do_add() -> None:
    input_str = input(ADD_PROMPT).strip()
    stop = False
    step_count = 1
    plan = []
    while not stop:
        plan_step = input(ADD_FOLLOWUP.format(n=step_count)).strip()
        if not plan_step:
            break
        plan.append(f"{step_count}. {plan_step}")
        step_count += 1
    plan_str = "\n".join(plan)
    sample_plan = SamplePlan(id=str(uuid4()), input=input_str, plan=plan_str)
    confirm = input(ADD_CONFIRM.format(plan=sample_plan.get_formatted_plan())).strip()
    if confirm == "y":
        db = get_psql()
        db.insert_sample_plan(sample_plan)
        print("added sample plan")


def do_list() -> None:
    db = get_psql()
    for sample_plan in db.get_all_sample_plans():
        print(f"{sample_plan.input} ({sample_plan.id})")


def do_delete() -> None:
    db = get_psql()
    delete_id = input(DELETE_PROMPT).strip()
    for sample_plan in db.get_all_sample_plans():
        if sample_plan.id == delete_id:
            confirm = input(DELETE_CONFIRM.format(plan=sample_plan.get_formatted_plan())).strip()
            if confirm == "y":
                db.delete_sample_plan(sample_plan.id)
                print("sample plan deleted")
            break


def do_show() -> None:
    db = get_psql()
    show_id = input(SHOW_PROMPT).strip()
    for sample_plan in db.get_all_sample_plans():
        if sample_plan.id == show_id:
            print(sample_plan.get_formatted_plan())


async def do_search() -> None:
    search_input = input(SEARCH_PROMPT).strip()
    similar_plans = await get_similar_sample_plans(search_input)
    for sample_plan in similar_plans:
        print(f"{sample_plan.input} ({sample_plan.id})")


async def main() -> None:
    while True:
        user_input = input(MAIN_PROMPT).strip()
        if user_input == "help":
            print(HELP)
        elif user_input == "exit":
            break
        elif user_input == "add":
            do_add()
        elif user_input == "list":
            do_list()
        elif user_input == "search":
            await do_search()
        elif user_input == "show":
            do_show()
        elif user_input == "delete":
            do_delete()


if __name__ == "__main__":
    asyncio.run(main())
