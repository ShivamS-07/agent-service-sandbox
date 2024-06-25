import asyncio
import json
from typing import List

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_types.text import ProfileText, Text, TextGroup, TopicProfiles
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.news import (
    GetNewsArticlesForTopicsInput,
    get_news_articles_for_topics,
)
from agent_service.tools.profiler.constants import (
    IMPORTANCE_POSTFIX,
    NEGATIVE,
    POSITIVE,
)
from agent_service.tools.profiler.helpers import write_profiles
from agent_service.tools.profiler.prompts import (
    GENERATE_PROFILE_DESCRIPTION,
    IMPACT_MAIN,
    IMPACT_SYS,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.string_utils import clean_to_json_if_needed


class GetCompanyProfilesForTopic(ToolArgs):
    topic: str
    relevant_text_data: List[Text]


@tool(
    description=GENERATE_PROFILE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
)
async def generate_profiles(
    args: GetCompanyProfilesForTopic, context: PlanRunContext
) -> TopicProfiles:
    llm = GPT(model=GPT4_O)

    text_group = TextGroup(val=args.relevant_text_data)
    texts_str: str = Text.get_all_strs(text_group, include_header=True, text_group_numbering=True)  # type: ignore

    # Brainstorm and generate impacts
    impacts_raw_output = await llm.do_chat_w_sys_prompt(
        main_prompt=IMPACT_MAIN.format(theme=args.topic, text_documents=texts_str),
        sys_prompt=IMPACT_SYS.format(),
    )

    impacts_json_list = json.loads(clean_to_json_if_needed(impacts_raw_output))

    profile_lookup = await write_profiles(
        theme=args.topic, impacts=impacts_json_list, news=texts_str
    )
    profile_texts: List[ProfileText] = []
    for profiles in profile_lookup.values():
        pos_profiles = profiles.get(POSITIVE, [])
        pos_profile_scores = profiles.get(f"{POSITIVE}{IMPORTANCE_POSTFIX}", [])
        if pos_profiles:
            for profile, score in zip(pos_profiles, pos_profile_scores):
                profile_texts.append(ProfileText(val=profile, importance_score=score))

        neg_profiles = profiles.get(NEGATIVE, [])
        neg_profile_scores = profiles.get(f"{NEGATIVE}{IMPORTANCE_POSTFIX}", [])
        if neg_profiles:
            for profile, score in zip(neg_profiles, neg_profile_scores):
                profile_texts.append(ProfileText(val=profile, importance_score=score))
    return TopicProfiles(topic=args.topic, val=profile_texts)


async def main() -> None:
    input_text = "Hello :)"
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])
    plan_context = PlanRunContext(
        agent_id="123",
        plan_id="123",
        user_id="123",
        plan_run_id="123",
        chat=chat_context,
        run_tasks_without_prefect=True,
        skip_db_commit=True,
    )

    wind_energy_articles = await get_news_articles_for_topics(
        GetNewsArticlesForTopicsInput(topics=["wind energy"]), context=plan_context
    )  # Find news articles related to wind energy

    profiles: TopicProfiles = await generate_profiles(
        GetCompanyProfilesForTopic(topic="wind energy", relevant_text_data=wind_energy_articles),  # type: ignore
        context=plan_context,
    )

    for profile in profiles.val:
        print(f"({profile.importance_score}) {profile.val}")


if __name__ == "__main__":
    asyncio.run(main())
