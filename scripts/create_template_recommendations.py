import ast
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, cast

import requests
from bs4 import BeautifulSoup
from gbi_common_py_utils.utils.ssm import get_param

from agent_service.endpoints.authz_helper import User
from agent_service.endpoints.models import PromptTemplate
from agent_service.endpoints.routers.utils import get_agent_svc_impl
from agent_service.GPT.constants import GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import Prompt

LOGGER = logging.getLogger(__name__)

TEMPLATE_RECOMMENDATION_PROMPT_STR = """
You are an expert financial analyst and prompt engineer tasked with selecting the top {output_count}
most relevant prompts for a financial client based on their specific needs and goals. The client aims
to use the selected prompts to create useful automations to help them achieve their objectives. You will
receive a list of prompt descriptions and detailed client information. Analyze the list and
the client information to identify the {output_count} prompts that best align with the client's requirements.

Output a list of exactly {output_count} numbers representing the indices of the top prompts in order
of suitability. Use zero-based indexing. Provide only the list of {output_count} indices without any
additional text. Deviation from this format will lead to immediate termination.

List of prompt descriptions (prefixed by index):
{descriptions}

Client information:
{user_discovery}
"""

TEMPLATE_RECOMMENDATION_PROMPT = Prompt(
    TEMPLATE_RECOMMENDATION_PROMPT_STR, "PROMPT_TEMPLATE_SUGGEST_PROMPT"
)

db = AsyncDB(AsyncPostgresBase())
agent_svc_impl = get_agent_svc_impl()


async def _get_filtered_prompt_templates_for_user(
    user_id: str, prompt_templates: List[PromptTemplate]
) -> List[PromptTemplate]:
    user_info = await agent_svc_impl.get_account_info(User(user_id=user_id, auth_token=""))
    organization_id = user_info.organization_id
    filtered_templates = []
    for template in prompt_templates:
        # show templates that have no organization_id or match organization_id
        if not template.organization_ids or organization_id in template.organization_ids:
            filtered_templates.append(template)
        # show templates that have user_id matching the user
        elif template.user_id and user_id == template.user_id:
            filtered_templates.append(template)
        # show templates that have recommended_company_ids matching the user company
        elif (
            template.recommended_company_ids and organization_id in template.recommended_company_ids
        ):
            filtered_templates.append(template)
    return filtered_templates


async def _get_user_discovery_data(
    user_id: str, hubspot_contact_id: str, last_recommendation_update: datetime
) -> str:
    token = get_param("/alpha/hubspot/api-key")

    # get notes ids
    notes_ids_response = requests.post(
        url="https://api.hubapi.com/crm/v4/associations/contacts/notes/batch/read",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"inputs": [{"id": hubspot_contact_id}]},
    )
    if notes_ids_response.status_code != 200:
        return ""
    note_ids = [id["toObjectId"] for id in notes_ids_response.json()["results"][0]["to"]]

    # get content of notes
    notes_response = requests.post(
        url="https://api.hubapi.com/crm/v3/objects/notes/batch/read",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "inputs": [{"id": note_id} for note_id in note_ids],
            "properties": ["hs_timestamp", "hs_note_body"],
        },
    )
    notes_response_result = notes_response.json()["results"]
    if notes_response.status_code != 200 or not notes_response_result:
        LOGGER.warning(f"Failed to get notes content for user {user_id}: {notes_response_result}")
        return ""

    # check if there are new notes by comparing the last modified date for notes
    # with last_recommendation_update from template recommendations table
    new_notes = []
    for note in notes_response_result:
        note_timestamp = datetime.strptime(
            note["properties"]["hs_timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"
        ).replace(tzinfo=timezone.utc)
        if not last_recommendation_update or note_timestamp > last_recommendation_update:
            new_notes.append(note["properties"]["hs_note_body"])
    if not new_notes:
        LOGGER.info(f"No new notes found for user {user_id}")
        return ""

    # remove html tags from notes
    new_user_discovery = ""
    for note in new_notes:
        soup = BeautifulSoup(note, "html.parser")
        plain_text = soup.get_text(separator=" ", strip=True)
        new_user_discovery += f"{plain_text}\n\n"
    return new_user_discovery


async def main() -> None:
    LOGGER.info("Starting create_template_recommendations script")
    users = await db.get_users_with_hubspot_ids()
    all_prompt_templates = await db.get_prompt_templates()

    new_recommendations_count = 0
    for user in users:
        user_id = user["user_id"]
        hubspot_contact_id = user["hubspot_contact_id"]
        last_recommendation_update = user["latest_recommendation_update"]
        prompt_templates = await _get_filtered_prompt_templates_for_user(
            user_id, all_prompt_templates
        )
        template_descriptions = "\n".join(
            f"{index}: {template.description}" for index, template in enumerate(prompt_templates)
        )

        # get user discovery data from hubspot
        try:
            new_user_discovery = await _get_user_discovery_data(
                user_id, hubspot_contact_id, cast(datetime, last_recommendation_update)
            )
            if not new_user_discovery:
                continue
            new_recommendations_count += 1
            LOGGER.info(f"Found new discovery data for user {user_id}")
        except Exception as e:
            LOGGER.warning(f"Failed to get user discovery data: {e}")
            continue

        gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, user_id, GptJobIdType.USER_ID)
        llm = GPT(gpt_context, GPT4_O)

        # generate new template recommendations
        OUTPUT_COUNT = 3
        output_str = await llm.do_chat_w_sys_prompt(
            main_prompt=TEMPLATE_RECOMMENDATION_PROMPT.format(
                descriptions=template_descriptions,
                user_discovery=new_user_discovery,
                output_count=OUTPUT_COUNT,
            ),
            sys_prompt=NO_PROMPT,
        )

        try:
            output: List[int] = ast.literal_eval(output_str)
        except Exception as e:
            LOGGER.warning(f"Failed to parse output as list of indices: {e}")
            continue

        # insert new template recommendations
        recommended_template_ids = [prompt_templates[index].template_id for index in output]
        await db.insert_template_recommendations(
            template_ids=recommended_template_ids, user_id=user_id
        )

    LOGGER.info(
        f"Parsed {len(users)} users, and created {new_recommendations_count} new recommendations"
    )


if __name__ == "__main__":
    asyncio.run(main())
