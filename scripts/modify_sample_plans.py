import asyncio
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import List
from uuid import uuid4

from agent_service.planner.planner_types import SamplePlan
from agent_service.planner.utils import get_similar_sample_plans
from agent_service.utils.enablement_function_registry import (
    is_plan_enabled,
    keyword_search_sample_plans,
)
from agent_service.utils.postgres import get_psql

GET_author_str = "Automatic username retrieval failed. What is your AWS username?> "

MAIN_PROMPT = "Choose an action (add/list/search/show/delete/modify/help/exit)> "

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
For information on creating a sample plan visit:
https://gradientboostedinvestments.atlassian.net/wiki/spaces/GBI/pages/2935324673/How+to+add+sample+plans
"""

ADD_PROMPT = "Enter the sample input> "
ADD_FOLLOWUP = "Enter Step {n} of the plan (enter nothing to stop) > "
ADD_CONFIRM = "Are you sure you want to add this sample plan:\n{plan}\n(y/n)> "


DELETE_PROMPT = "Enter the sample plan id you want to delete> "
SOFT_DELETE_PROMPT = (
    "Would you like to disable the plan instead of deleting it?\nEnter y to disable> "  # noqa
)
DELETE_CONFIRM = "Are you sure you want to delete this sample plan:\n{plan}\n(y/n)> "
SOFT_DELETE_CONFIRM = "Are you sure you want to disable this sample plan:\n{plan}\n(y/n)> "
DELETED_RESPONSE = "Sample plan deleted"
DISABLED_RESPONSE = "Sample plan disabled"

SHOW_PROMPT = "Enter the sample plan id of the plan you want to inspect> "

SEARCH_PROMPT = "Enter an input that you want to find similar inputs to> "

GET_ID = "Enter the ID of the sample plan you wish to modify> "
SHOW_MODIFY_OPTIONS = "Choose what part of the sample plan to modify (plan/relevance/input/category/note/enable/disable/enablement)> "  # noqa
PLAN = "plan"
PLAN_DOES_NOT_EXIST = "This sample plan does not exist!"
GET_PLAN_OPTIONS = "Choose the plan modification (insert/delete/modify)> "
CHANGE = "modify"
ADD_LINE = "insert"
DELETE_LINE = "delete"
ENABLE = "enable"
DISABLE = "disable"
GET_LINE_NUMBER = "Enter the number of the line you wish to modify> "
GET_NEW_LINE = "Enter the line you would like to add> "
GET_DELETE_LINE = "Enter the number of the line you wish to delete> "
RELEVANCE = "relevance"
INPUT = "input"
FLAG = "flag"
NOTE = "note"
GET_NEW_NOTE = "Enter the note you'd like to add> "
REPLACE_CONFIRM = "Confirm you'd like to replace:\n{old}\nwith:\n{new}\nEnter y to confirm> "
ADD_LINE_CONFIRM = "Confirm you'd like to add the following line:\n{add_line}\nAt position: {pos}\nEnter y to confirm> "  # noqa
DELETE_LINE_CONFIRM = "Confirm you'd like to delete:\n{old}\nEnter y to confirm> "
REPLACE_PLAN_STEPS_CONFIRM = (
    "Confirm you'd like to replace the old plan with the following:\n{new}\nEnter y to confirm> "
)
MODIFIED_PLAN_MESSAGE = "Successfully modified plan! (ID: {sample_plan_id})"
RELEVANCE_INPUT = "Enter the new relevance value> "
PROMPT_INPUT = "Enter the new plan input> "
FLAG_INPUT = "Enter the new flag name> "
CATEGORY_INPUT = "Enter the new category name> "
CATEGORY = "category"
ENABLED_INPUT = "Enter the new enablement value> "
ENABLED = "enablement"
DISCARDING_CHANGES = "Discarding changes..."
ADD_VALUE_CONFIRM = "Confirm you'd like to add the following value:\n{value}\nEnter y to confirm> "
DISPLAY_PLAN_LIST = "Here is the updated plan list: "
YOU_HAVE_CHOSEN_LINE = "You have chosen the following line: "
LINE_BY_LINE = (
    "Enter y to modify line-by-line, all other input opens an editor containing the whole plan> "  # noqa
)
VIM = "vim"
ENV_EDITOR = "EDITOR"
DEFAULT_EDITOR_FILE = "modify_sample_plans_editor.txt"
NOTE_CHANGE_STRING = "\nThe above note was added by {author_str} on {today}\n\n"
ADD_NOTE_NON_EMPTY = "Add a note. Must be non-empty> "
ALLOW_EMPTY_INPUT = "The following value can be empty (press enter to skip it).\n"
SKIPPING = "Skipping..."
CONFIRM_ENABLE_OR_DISABLE = "Enter y to confirm that you would like to {choice} the sample plan> "
NOTE_INVALID_REENTER = "Note is invalid. Please enter again."
NOTE_INVALID_PROMPT = """The note was invalid because it did not contain any of the following:
A slack URL, a Jira ticket ID or URL, a GitHub URL, an Agent URL, or the word "example" explicitly.
Would you like to re-enter the note, or skip this warning?
Enter y to skip this warning> """
QUALITY_POD_REGEX_PATTERN = r"ql\d{2}-\d+"
MEMORY_POD_REGEX_PATTERN = r"mm\d{2}-\d+"
CORE_POD_REGEX_PATTERN = r"wc-\d+"
SLACK_URL = "gbiworkspace.slack.com"
JIRA_URL = "gradientboostedinvestments.atlassian.net"
AGENT_URL = ".boosted.ai"
GITHUB_URL = "github.com"
EXAMPLE_STRING = "example"
INVALID_LINE_NUMBER = "Invalid line number entered!"


def do_add(author_str: str) -> None:
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
    note_str = ""
    while len(note_str) == 0:
        note_str = input(ADD_NOTE_NON_EMPTY).strip()
    note_str += NOTE_CHANGE_STRING.format(
        author_str=author_str, today=datetime.datetime.now().isoformat()
    )
    enablement_str = input(ALLOW_EMPTY_INPUT + ENABLED_INPUT).strip()
    id = str(uuid4())
    sample_plan = SamplePlan(
        id=id,
        input=input_str,
        plan=plan_str,
        author=author_str,
        last_updated_author=author_str,
        note=note_str,
        relevance=None,
        category=None,
        enabled=enablement_str,
        changelog="Plan initially created",
    )
    confirm = input(ADD_CONFIRM.format(plan=sample_plan.get_formatted_plan_internal())).strip()
    print(f"Plan enablement status: {is_plan_enabled(sample_plan)}")
    if confirm.lower() == "y":
        db = get_psql()
        db.insert_sample_plan(sample_plan)
        print(f"added sample plan: {id=}")


def display_plan_list(plan_lst: List[str]) -> None:
    for i, step in enumerate(plan_lst):
        print(str(i + 1) + ". " + step)


def modify_text_in_editor(initial_text: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_file.write(initial_text)
        temp_file.flush()
        temp_file_name = temp_file.name

    try:
        if sys.platform == "win32":
            editor = os.getenv("EDITOR", "notepad")
        else:
            editor = os.getenv("EDITOR", "vim")
        subprocess.run([editor, temp_file_name])

        with open(temp_file_name, "r") as temp_file:
            updated_text = temp_file.read()
    finally:
        os.remove(temp_file_name)

    return updated_text


def do_modify(author_str: str) -> None:
    input_id = input(GET_ID).strip()
    pg = get_psql()
    plan = pg.get_sample_plan_by_id(input_id)
    if not plan:
        return None
    plan_lst = []
    outer_stop = outer_changed = False
    lines = plan.plan.splitlines()
    print(plan.get_formatted_plan_internal())
    for idx, line in enumerate(lines):
        modified_line = line[len(str(idx + 1)) + 1 :].strip()
        # the above removes the number prefixes at the start of the steps

        plan_lst.append(modified_line)
    while not outer_stop:
        what_to_modify = input(SHOW_MODIFY_OPTIONS).strip()
        stop = False

        if what_to_modify == PLAN:
            line_by_line = input(LINE_BY_LINE).strip()
            if line_by_line == "y" or line_by_line == "<y>":
                changed = False
                while not stop:
                    input_str = input(GET_PLAN_OPTIONS).strip()

                    if input_str == CHANGE:
                        line_number = int(input(GET_LINE_NUMBER).strip())
                        if line_number - 1 > len(plan_lst) or line_number <= 0:
                            break
                        else:
                            print(YOU_HAVE_CHOSEN_LINE)
                            print(plan_lst[line_number - 1])
                        new_line = input(GET_NEW_LINE).strip()
                        confirm = input(
                            REPLACE_CONFIRM.format(old=plan_lst[line_number - 1], new=new_line)
                        ).strip()
                        if confirm.lower() == "y":
                            plan_lst[line_number - 1] = new_line
                            changed = outer_changed = True
                            print(DISPLAY_PLAN_LIST)
                            display_plan_list(plan_lst)
                        else:
                            print(DISCARDING_CHANGES)

                    elif input_str == ADD_LINE:
                        line_pos = int(input(GET_LINE_NUMBER).strip())
                        if line_pos > len(plan_lst) + 1 or line_pos <= 0:
                            break
                        new_line = input(GET_NEW_LINE).strip()
                        confirm = input(
                            ADD_LINE_CONFIRM.format(add_line=new_line, pos=line_pos)
                        ).strip()
                        if confirm.lower() == "y":
                            plan_lst.insert(line_pos - 1, new_line)
                            changed = outer_changed = True
                            print(DISPLAY_PLAN_LIST)
                            display_plan_list(plan_lst)
                        else:
                            print(DISCARDING_CHANGES)

                    elif input_str == DELETE_LINE:
                        line_pos = int(input(GET_DELETE_LINE).strip())
                        if line_pos > len(plan_lst) or line_pos <= 0:
                            print(INVALID_LINE_NUMBER)
                            break
                        confirm = input(
                            DELETE_LINE_CONFIRM.format(old=plan_lst[line_pos - 1])
                        ).strip()
                        if confirm.lower() == "y":
                            plan_lst.pop(line_pos - 1)
                            changed = outer_changed = True
                            print(DISPLAY_PLAN_LIST)
                            display_plan_list(plan_lst)
                        else:
                            print(DISCARDING_CHANGES)

                    else:
                        if changed:
                            new_plan_str = "\n".join(
                                [str(str(i + 1) + ". " + step) for i, step in enumerate(plan_lst)]
                            )
                            confirm = input(
                                REPLACE_PLAN_STEPS_CONFIRM.format(new=new_plan_str)
                            ).strip()
                            if confirm.lower() == "y":
                                get_psql().modify_sample_plan(
                                    input_id, last_updated_author=author_str, plan=new_plan_str
                                )
                                print(MODIFIED_PLAN_MESSAGE.format(sample_plan_id=input_id))
                            else:
                                print(DISCARDING_CHANGES)
                        stop = True

            else:
                updated_plan = modify_text_in_editor(
                    "\n".join([str(str(i + 1) + ". " + step) for i, step in enumerate(plan_lst)])
                )
                confirm = input(REPLACE_PLAN_STEPS_CONFIRM.format(new=updated_plan)).strip()
                if confirm.lower() == "y":
                    plan_lst = [
                        line[len(str(idx + 1)) + 1 :].strip()
                        for idx, line in enumerate(updated_plan.splitlines())
                    ]
                    get_psql().modify_sample_plan(
                        input_id, last_updated_author=author_str, plan=updated_plan
                    )
                    print(MODIFIED_PLAN_MESSAGE.format(sample_plan_id=input_id))
                else:
                    print(DISCARDING_CHANGES)

        elif what_to_modify == RELEVANCE:
            while not stop:
                updated_relevance = float(input(RELEVANCE_INPUT).strip())
                if plan.relevance:
                    confirm = input(
                        REPLACE_CONFIRM.format(old=plan.relevance, new=updated_relevance)
                    ).strip()
                    if confirm.lower() == "y":
                        get_psql().modify_sample_plan(
                            input_id, last_updated_author=author_str, relevance=updated_relevance
                        )
                        outer_changed = True
                        stop = True
                    else:
                        print(DISCARDING_CHANGES)
                else:
                    confirm = input(ADD_VALUE_CONFIRM.format(value=updated_relevance)).strip()
                    if confirm.lower() == "y":
                        get_psql().modify_sample_plan(
                            input_id, last_updated_author=author_str, relevance=updated_relevance
                        )
                        outer_changed = True
                        stop = True
                    else:
                        print(DISCARDING_CHANGES)

        elif what_to_modify == INPUT:
            while not stop:
                new_prompt = input(PROMPT_INPUT).strip()
                confirm = input(REPLACE_CONFIRM.format(old=plan.input, new=new_prompt)).strip()
                if confirm.lower() == "y":
                    get_psql().modify_sample_plan(
                        input_id, last_updated_author=author_str, input=new_prompt
                    )
                    outer_changed = True
                    stop = True
                else:
                    print(DISCARDING_CHANGES)

        elif what_to_modify == ENABLE:
            confirm = input(CONFIRM_ENABLE_OR_DISABLE.format(choice=ENABLE)).strip()
            if confirm.lower() == "y":
                get_psql().enable_sample_plan(input_id)
                print("Plan enablement status: True")
            else:
                print(SKIPPING)

        elif what_to_modify == DISABLE:
            confirm = input(CONFIRM_ENABLE_OR_DISABLE.format(choice=DISABLE)).strip()
            if confirm.lower() == "y":
                get_psql().disable_sample_plan(input_id)
                print("Plan enablement status: False")
            else:
                print(SKIPPING)

        elif what_to_modify == CATEGORY:
            while not stop:
                new_category = input(CATEGORY_INPUT).strip()
                if plan.category:
                    confirm = input(
                        REPLACE_CONFIRM.format(old=plan.category, new=new_category)
                    ).strip()
                    if confirm.lower() == "y":
                        get_psql().modify_sample_plan(
                            input_id, last_updated_author=author_str, category=new_category
                        )
                        plan.category = new_category
                        outer_changed = True
                        stop = True
                    else:
                        print(DISCARDING_CHANGES)
                else:
                    confirm = input(ADD_VALUE_CONFIRM.format(value=new_category)).strip()
                    if confirm.lower() == "y":
                        get_psql().modify_sample_plan(
                            input_id, last_updated_author=author_str, category=new_category
                        )
                        plan.category = new_category
                        outer_changed = True
                        stop = True
                    else:
                        print(DISCARDING_CHANGES)

        elif what_to_modify == ENABLED:
            while not stop:
                new_enabled_str = input(ENABLED_INPUT).strip()
                if plan.enabled:
                    confirm = input(
                        REPLACE_CONFIRM.format(old=plan.enabled, new=new_enabled_str)
                    ).strip()
                    if confirm.lower() == "y":
                        get_psql().modify_sample_plan(
                            input_id, last_updated_author=author_str, enabled=new_enabled_str
                        )
                        plan.enabled = new_enabled_str
                        print(f"Plan enablement status: {is_plan_enabled(plan)}")
                        outer_changed = True
                        stop = True
                    else:
                        print(DISCARDING_CHANGES)
                else:
                    confirm = input(ADD_VALUE_CONFIRM.format(value=new_enabled_str)).strip()
                    if confirm.lower() == "y":
                        get_psql().modify_sample_plan(
                            input_id, last_updated_author=author_str, enabled=new_enabled_str
                        )
                        plan.enabled = new_enabled_str
                        print(f"Plan enablement status: {is_plan_enabled(plan)}")
                        outer_changed = True
                        stop = True
                    else:
                        print(DISCARDING_CHANGES)

        elif what_to_modify == NOTE:
            validate_note = False
            new_note = plan.note if plan.note else ""
            while not validate_note:
                new_note = modify_text_in_editor(new_note)
                new_note_lower = new_note.lower()
                if EXAMPLE_STRING in new_note_lower:
                    validate_note = True
                elif GITHUB_URL in new_note_lower:
                    validate_note = True
                elif AGENT_URL in new_note_lower:
                    validate_note = True
                elif JIRA_URL in new_note_lower:
                    validate_note = True
                elif SLACK_URL in new_note_lower:
                    validate_note = True
                elif bool(re.search(QUALITY_POD_REGEX_PATTERN, new_note_lower)):
                    validate_note = True
                elif bool(re.search(MEMORY_POD_REGEX_PATTERN, new_note_lower)):
                    validate_note = True
                elif bool(re.search(CORE_POD_REGEX_PATTERN, new_note_lower)):
                    validate_note = True
                else:
                    skip_error = input(NOTE_INVALID_PROMPT)
                    if skip_error == "y":
                        validate_note = True
                    else:
                        print(NOTE_INVALID_REENTER)

            new_note += NOTE_CHANGE_STRING.format(
                author_str=author_str, today=datetime.datetime.now().isoformat()
            )

            if plan.note:
                confirm = input(REPLACE_CONFIRM.format(old=plan.note, new=new_note)).strip()
                if confirm.lower() == "y":
                    get_psql().modify_sample_plan(
                        input_id, last_updated_author=author_str, note=new_note
                    )
                    outer_changed = True
                else:
                    print(DISCARDING_CHANGES)
            else:
                confirm = input(ADD_VALUE_CONFIRM.format(value=new_note)).strip()
                if confirm.lower() == "y":
                    get_psql().modify_sample_plan(
                        input_id, last_updated_author=author_str, note=new_note
                    )
                    outer_changed = True
                else:
                    print(DISCARDING_CHANGES)

        else:
            outer_stop = True

    if outer_changed:
        print("Finished sample plan:")
        finished_plan = get_psql().get_sample_plan_by_id(input_id)
        if finished_plan is not None:
            print(plan.get_formatted_plan_internal())
            print(f"Plan enablement status: {is_plan_enabled(finished_plan)}")


def do_list() -> None:
    db = get_psql()
    print(
        [
            "\n\n".join(sample_plan.get_formatted_plan_internal())
            for sample_plan in db.get_all_sample_plans()
        ]
    )


def do_delete() -> None:
    db = get_psql()
    delete_id = input(DELETE_PROMPT).strip()
    sample_plan = get_psql().get_sample_plan_by_id(delete_id)
    if sample_plan:
        if sample_plan.id == delete_id:
            soft_delete = input(SOFT_DELETE_PROMPT).strip()
            if soft_delete == "y":
                confirm = input(
                    SOFT_DELETE_CONFIRM.format(plan=sample_plan.get_formatted_plan_internal())
                ).strip()
                if confirm.lower() == "y":
                    db.disable_sample_plan(sample_plan.id)
                    print(DISABLED_RESPONSE)
            else:
                confirm = input(
                    DELETE_CONFIRM.format(plan=sample_plan.get_formatted_plan_internal())
                ).strip()
                if confirm.lower() == "y":
                    db.delete_sample_plan(sample_plan.id)
                    print(DELETED_RESPONSE)
    else:
        print(PLAN_DOES_NOT_EXIST)


def do_show() -> None:
    show_id = input(SHOW_PROMPT).strip()
    plan = get_psql().get_sample_plan_by_id(show_id)
    if plan:
        print(plan.get_formatted_plan_internal(show_changelog=True))
    else:
        print(PLAN_DOES_NOT_EXIST)


async def do_search() -> None:
    search_input = input(SEARCH_PROMPT).strip()
    if len(" ".split(search_input)) < 2:
        similar_plans = keyword_search_sample_plans(search_input)
        for sample_plan in similar_plans:
            print(f"{sample_plan.input} ({sample_plan.id})")
    else:
        similar_plans = await get_similar_sample_plans(search_input)
        for sample_plan in similar_plans:
            print(f"{sample_plan.input} ({sample_plan.id})")


async def main() -> None:
    try:
        cmd = "aws sts get-caller-identity; exit 0"
        if sys.platform == "win32":
            cmd = "aws sts get-caller-identity && exit 0"

        author_str_bytes = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        """
        converting author_str_bytes to a dict should look like:
        {
            "UserId": ID here,
            "Account": account number,
            "Arn": "arn:aws:iam::ACCOUNT NUMBER:user/AUTHOR_STR"
        }
        """
        author_str: str = json.loads(author_str_bytes)["Arn"].split("/")[-1]
    except subprocess.CalledProcessError:
        author_str = input(GET_author_str).strip()
    except KeyError:
        author_str = input(GET_author_str).strip()
    while True:
        user_input = input(MAIN_PROMPT).strip()
        if user_input == "help":
            print(HELP)
        elif user_input == "exit":
            break
        elif user_input == "add":
            do_add(author_str)
        elif user_input == "list":
            do_list()
        elif user_input == "search":
            await do_search()
        elif user_input == "show":
            do_show()
        elif user_input == "delete":
            do_delete()
        elif user_input == "modify":
            do_modify(author_str)


if __name__ == "__main__":
    asyncio.run(main())
