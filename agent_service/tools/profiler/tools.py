import asyncio
import json
from typing import List, Optional

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_types.idea import Idea
from agent_service.io_types.text import ProfileText, Text, TextGroup, TopicProfiles
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.ideas.utils import ideas_enabled
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
    PER_IDEA_GENERATE_PROFILE_DESCRIPTION,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.string_utils import clean_to_json_if_needed
from agent_service.utils.tool_diff import get_prev_run_info


async def get_profiles(
    topic: str, texts_str: str, context: PlanRunContext, idea: Optional[Idea] = None
) -> TopicProfiles:
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(model=GPT4_O, context=gpt_context)

    # Brainstorm and generate impacts
    chopped_texts_str = GPTTokenizer(model=llm.model).do_truncation_if_needed(
        truncate_str=texts_str,
        other_prompt_strs=[
            IMPACT_MAIN.template,
            IMPACT_SYS.template,
            topic,
        ],
    )

    impacts_raw_output = await llm.do_chat_w_sys_prompt(
        main_prompt=IMPACT_MAIN.format(theme=topic, text_documents=chopped_texts_str),
        sys_prompt=IMPACT_SYS.format(),
    )

    impacts_json_list = json.loads(clean_to_json_if_needed(impacts_raw_output))

    profile_lookup = await write_profiles(theme=topic, impacts=impacts_json_list, news=texts_str)
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

    if idea:
        return TopicProfiles(topic=topic, val=profile_texts, initial_idea=idea.title)
    else:
        return TopicProfiles(topic=topic, val=profile_texts)


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
    text_group = TextGroup(val=args.relevant_text_data)
    texts_str: str = await Text.get_all_strs(text_group, include_header=True, text_group_numbering=True)  # type: ignore

    return await get_profiles(args.topic, texts_str, context)


class PerIdeaGetCompanyProfilesForTopic(ToolArgs):
    ideas: List[Idea]
    topic_template: str


@tool(
    description=PER_IDEA_GENERATE_PROFILE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
    enabled=True,
    enabled_checker_func=ideas_enabled,
)
async def per_idea_generate_profiles(
    args: PerIdeaGetCompanyProfilesForTopic, context: PlanRunContext
) -> List[TopicProfiles]:
    logger = get_prefect_logger(__name__)

    # TODO: Remove once summarizer tool is merged
    IDEA = "IDEA"

    existing_profile_lookup = {}

    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "per_idea_generate_profiles")
        if prev_run_info is not None:
            prev_args = PerIdeaGetCompanyProfilesForTopic.model_validate_json(
                prev_run_info.inputs_str
            )
            prev_output: List[TopicProfiles] = prev_run_info.output  # type:ignore
            old_profile_lookup = {profile.initial_idea: profile for profile in prev_output}
            existing_profile_lookup = {
                idea.title: old_profile_lookup[idea.title]
                for idea in prev_args.ideas
                if idea.title in old_profile_lookup
            }

    except Exception as e:
        logger.warning(
            f"Failed attempt to update from previous iteration due to {e}, from scratch fallback"
        )

    topic_profiles_for_ideas: List[TopicProfiles] = []
    tasks = []

    existing_profile_count = 0
    for idea in args.ideas:
        if idea.title in existing_profile_lookup:
            tasks.append(identity(existing_profile_lookup[idea.title]))
            existing_profile_count += 1
            continue
        try:
            related_news_texts = await get_news_articles_for_topics(
                GetNewsArticlesForTopicsInput(topics=[idea.title]), context=context
            )

            text_group = TextGroup(val=related_news_texts)  # type: ignore
            texts_str: str = await Text.get_all_strs(
                text_group, include_header=True, text_group_numbering=True
            )  # type: ignore

            topic = args.topic_template.replace(IDEA, idea.title)
            tasks.append(get_profiles(topic, texts_str, context, idea))
        except EmptyOutputError:
            logger.warning(f"Couldn't generate profiles for idea: {idea.title}")

    topic_profiles_for_ideas = await gather_with_concurrency(tasks, n=10)
    if existing_profile_count > 0:
        await tool_log(f"Using existing profiles for {existing_profile_count} ideas", context)
    return topic_profiles_for_ideas


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
