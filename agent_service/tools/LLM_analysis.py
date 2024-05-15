import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple

from agent_service.GPT.constants import DEFAULT_SMART_MODEL, FILTER_CONCURRENCY
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import IOType
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.dates import DateFromDateStrInput, get_date_from_date_str
from agent_service.tools.lists import CollapseListsInput, collapse_lists
from agent_service.tools.stock_identifier_lookup import (
    StockIdentifierLookupInput,
    stock_identifier_lookup,
)
from agent_service.tools.stock_news import (
    GetNewsDevelopmentDescriptionsInput,
    GetNewsDevelopmentsAboutCompaniesInput,
    get_news_development_descriptions,
    get_news_developments_about_companies,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import Prompt

# PROMPTS

SUMMARIZE_SYS_PROMPT = Prompt(
    name="LLM_SUMMARIZE_SYS_PROMPT",
    template="You are a financial analyst tasked with summarizing one or more texts according to the instructions of an important client. You will be provided with the texts as well as transcript of your conversation with the client. If the client has provided you with any specifics about the format or content of the summary, you must follow those instructions. Otherwise, you should write a normal prose summary that touches on what you be to be the most important points that you see across all the text you have been provided on. The most important points are those which are highlighted, repeated, or otherwise appear most relevant to the user's expressed interest, if any. If none of these criteria seem to apply, use your best judgment on what seems to be important. Unless the user says otherwise, your output should be must smaller (a small fraction) of all the text provided. For example, if the input is involves several news summaries, a single sentence or two would be appropriate. Individual texts in your collection are delimited by ***",  # noqa: E501
)

SUMMARIZE_PROMPT_MAIN = Prompt(
    name="LLM_SUMMARIZE_MAIN_PROMPT",
    template="Summarize the following text(s) based on the needs of the client. Here are the documents, delimited by -----:\n-----\n{texts}\n-----\nHere is the transcript of your interaction with the client, delimited by ----:\n----\n{chat_context}\n----\nNow write your summary",  # noqa: E501
)

TOPIC_FILTER_SYS_PROMPT = Prompt(
    name="TOPIC_FILTER_SYS_PROMPT",
    template="You are a financial analyst checking a text to see if it is relevant to the provided topic. On the first line of your output, if you think there is at least some relevance to the topic, please briefly discuss the nature of relevance in no more than 30 words. If there is absolutely no relevance, you should simply output `No relevance`. Then on the second line, output a number between 0 and 3. 0 indicates no relevance, 1 indicates some relevance, 2 indicates moderate relevance, and 3 should be used when the text is clearly highly relevant to the topic. Most of the texts you will read will not be relevant, and so 0 should be your default.",  # noqa: E501
)

TOPIC_FILTER_MAIN_PROMPT = Prompt(
    name="TOPIC_FILTER_MAIN_PROMPT",
    template="Decide to what degree the following text is relevant to the provided topic. Here is the text, delimited by ---:\n---\n{text}\n---\n. The topic is: {topic}. Write your discussion, followed by your relevant rating between 0 and 3: ",  # noqa: E501
)


# These are to try to force the filter to allow some hits but not too many
LLM_FILTER_MAX_PERCENT = 0.2
LLM_FILTER_MIN_PERCENT = 0.05


class SummarizeTextInput(ToolArgs):
    texts: List[str]


@tool(
    description=(
        "This function takes a list of texts and uses an LLM to summarize them into a single text "
        "based on the instructions provided by the user in their input. Note: before you run this"
        " function you must make sure to apply all relevant filters on the texts, do not use "
        " this function to filter large quantities of text"
    ),
    category=ToolCategory.LLM_ANALYSIS,
)
async def summarize_texts(args: SummarizeTextInput, context: PlanRunContext) -> str:
    if context.chat is None:
        return ""
    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_PLANNER, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_SMART_MODEL)
    result = await llm.do_chat_w_sys_prompt(
        SUMMARIZE_PROMPT_MAIN.format(
            texts="\n***\n".join(args.texts), chat_context=context.chat.get_gpt_input()
        ),
        SUMMARIZE_SYS_PROMPT.format(),
    )
    return result


async def topic_filter_helper(
    texts: List[str], topic: str, agent_id: str
) -> List[Tuple[bool, str]]:
    gpt_context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=DEFAULT_SMART_MODEL)
    tasks = []
    for text in texts:
        tasks.append(
            llm.do_chat_w_sys_prompt(
                TOPIC_FILTER_MAIN_PROMPT.format(text=text, topic=topic),
                TOPIC_FILTER_SYS_PROMPT.format(),
            )
        )
    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)

    counts: Dict[int, int] = defaultdict(int)
    score_tuples = []
    for result in results:
        try:
            rationale, score = result.strip().split("\n")
            score = int(score)
        except ValueError:
            score = 0
            rationale = "No relevance"
        counts[score] += 1
        score_tuples.append((score, rationale))
    if counts[3] > len(texts) * LLM_FILTER_MAX_PERCENT:
        # If there are lot of 3s, only include 3
        cutoff = 3
    elif counts[3] + counts[2] < len(text) * LLM_FILTER_MAX_PERCENT:
        # If there are hardly any 3 + 2, include 1s
        cutoff = 1
    else:
        # otherwise, includes 2s and 3s
        cutoff = 2

    return [(score >= cutoff, rationale) for score, rationale in score_tuples]


class FilterTextsByTopicInput(ToolArgs):
    topic: str
    texts: List[str]


@tool(
    description=(
        "This function takes a topic and list of texts and uses an LLM to filter the list to only those"
        " that are relevant to the provided topic. Can be applied to news, earnings, SEC filings, and any"
        " other text. "
        " It is better to call this function once with a complex topic with many ideas than to call this"
        " function many times with smaller topics. Use filter_items_by_topic if you have things other "
        "than texts that you want to filter"
    ),
    category=ToolCategory.LLM_ANALYSIS,
)
async def filter_texts_by_topic(
    args: FilterTextsByTopicInput, context: PlanRunContext
) -> List[str]:
    # not currently returning rationale, but will probably want it
    return [
        text
        for text, (is_relevant, _) in zip(
            args.texts, await topic_filter_helper(args.texts, args.topic, context.agent_id)
        )
        if is_relevant
    ]


class FilterItemsByTopicInput(ToolArgs):
    topic: str
    items: List[IOType]
    texts: List[str]


@tool(
    description=(
        "This function takes any list of items (of any kind) which has some corresponding associated texts"
        " and uses an LLM to filter to only those objects whose text representation is relevant to the provided topic."
    ),
    category=ToolCategory.LLM_ANALYSIS,
)
async def filter_items_by_topic(
    args: FilterItemsByTopicInput, context: PlanRunContext
) -> List[IOType]:
    return [
        item
        for item, (is_relevant, _) in zip(
            args.items, await topic_filter_helper(args.texts, args.topic, context.agent_id)
        )
        if is_relevant
    ]


async def main() -> None:
    input_text = "Can you give me a single summary of news published in the last week about machine learning at Meta, Apple, and Microsoft?"  # noqa: E501
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])
    plan_context = PlanRunContext(
        agent_id="123",
        plan_id="123",
        user_id="123",
        plan_run_id="123",
        chat=chat_context,
        run_tasks_without_prefect=True,
    )
    start_date = await get_date_from_date_str(
        DateFromDateStrInput(date_str="1 week ago"), plan_context
    )  # Get the date for one month ago
    print(start_date)
    stock_ids = [
        await stock_identifier_lookup(StockIdentifierLookupInput(stock_str=stock), plan_context)
        for stock in ["Meta", "Apple", "Microsoft"]
    ]  # Convert stock names to identifiers
    print(stock_ids)
    news_developments = await get_news_developments_about_companies(
        GetNewsDevelopmentsAboutCompaniesInput(stock_ids=stock_ids, start_date=start_date),  # type: ignore
        plan_context,
    )  # Get news developments for the last month for Meta, Apple, and Microsoft
    print(len(news_developments[0]))  # type: ignore
    print(len(news_developments[1]))  # type: ignore
    print(len(news_developments[2]))  # type: ignore
    collapsed_news_ids = await collapse_lists(
        CollapseListsInput(lists_of_lists=news_developments), plan_context  # type: ignore
    )  # Collapse the list of lists of news ids into a single list
    print(len(collapsed_news_ids))  # type: ignore
    news_descriptions = await get_news_development_descriptions(
        GetNewsDevelopmentDescriptionsInput(development_ids=collapsed_news_ids), plan_context  # type: ignore
    )  # Retrieve the text descriptions of the news developments
    filtered_news = await filter_texts_by_topic(
        FilterTextsByTopicInput(topic="machine learning", texts=news_descriptions), plan_context  # type: ignore
    )  # Filter the news descriptions to only those relevant to machine learning
    print(len(filtered_news))  # type: ignore
    print(filtered_news[0])  # type: ignore
    summary = await summarize_texts(
        SummarizeTextInput(texts=filtered_news), plan_context  # type: ignore
    )  # Summarize the filtered news texts into a single summary
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
