import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple

from agent_service.GPT.constants import DEFAULT_SMART_MODEL, FILTER_CONCURRENCY
from agent_service.GPT.requests import GPT
from agent_service.io_types.text import StockAlignedTextGroups, Text
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.dates import DateFromDateStrInput, get_date_from_date_str
from agent_service.tools.news import (
    GetNewsDevelopmentsAboutCompaniesInput,
    get_all_news_developments_about_companies,
    get_stock_aligned_news_developments,
)
from agent_service.tools.stocks import (
    StockIdentifierLookupInput,
    stock_identifier_lookup,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import Prompt

# PROMPTS

SUMMARIZE_SYS_PROMPT = Prompt(
    name="LLM_SUMMARIZE_SYS_PROMPT",
    template="You are a financial analyst tasked with summarizing one or more texts according to the instructions of an important client. You will be provided with the texts as well as transcript of your conversation with the client. If the client has provided you with any specifics about the format or content of the summary, you must follow those instructions. If a specific topic is mentioned, you must only include information about that topic. Otherwise, you should write a normal prose summary that touches on what you be to be the most important points that you see across all the text you have been provided on. The most important points are those which are highlighted, repeated, or otherwise appear most relevant to the user's expressed interest, if any. If none of these criteria seem to apply, use your best judgment on what seems to be important. Unless the user says otherwise, your output should be must smaller (a small fraction) of all the text provided. For example, if the input is involves several news summaries, a single sentence or two would be appropriate. Individual texts in your collection are delimited by ***",  # noqa: E501
)

SUMMARIZE_PROMPT_MAIN = Prompt(
    name="LLM_SUMMARIZE_MAIN_PROMPT",
    template="Summarize the following text(s) based on the needs of the client. Here are the documents, delimited by -----:\n-----\n{texts}\n-----\nHere is the transcript of your interaction with the client, delimited by ----:\n----\n{chat_context}\n----\nNow write your summary",  # noqa: E501
)

TOPIC_FILTER_SYS_PROMPT = Prompt(
    name="TOPIC_FILTER_SYS_PROMPT",
    template="You are a financial analyst checking a text or collection of texts to see if there is anything in the texts that is strongly relevant to the provided topic. On the first line of your output, if you think there is at least some relevance to the topic, please briefly discuss the nature of relevance in no more than 30 words. If there is absolutely no relevance, you should simply output `No relevance`. Then on the second line, output a number between 0 and 3. 0 indicates no relevance, 1 indicates some relevance, 2 indicates moderate relevance, and 3 should be used when the text is clearly highly relevant to the topic. Most of the texts you will read will not be relevant, and so 0 should be your default.",  # noqa: E501
)

TOPIC_FILTER_MAIN_PROMPT = Prompt(
    name="TOPIC_FILTER_MAIN_PROMPT",
    template="Decide to what degree the following text or texts have information that is relevant to the provided topic. Here is the text or texts, delimited by ---:\n---\n{text}\n---\n. The topic is: {topic}. Write your discussion, followed by your relevant rating between 0 and 3: ",  # noqa: E501
)


# These are to try to force the filter to allow some hits but not too many
LLM_FILTER_MAX_PERCENT = 0.2
LLM_FILTER_MIN_PERCENT = 0.05


class CombineStockAlignedTextGroupsInput(ToolArgs):
    text_groups1: StockAlignedTextGroups
    text_groups2: StockAlignedTextGroups


@tool(
    description=(
        "This function combines two StockAlignedTextGroups by joining the paired TextGroups "
        "across the two mappings, creating a single mapping to the combined TextGroups. "
        "Use this function when combining the output of different data retrieval functions called over the same ids. "
        "In particular, if you want to apply filter or search and the request mentions two different "
        "data sources, you should combine using this function before calling the LLM."
    ),
    category=ToolCategory.LLM_ANALYSIS,
    is_visible=False,
)
async def combine_stock_aligned_text_groups(
    args: CombineStockAlignedTextGroupsInput, context: PlanRunContext
) -> StockAlignedTextGroups:
    return StockAlignedTextGroups.join(args.text_groups1, args.text_groups2)


class SummarizeTextInput(ToolArgs):
    texts: List[Text]


@tool(
    description=(
        "This function takes a list of Texts of any kind and uses an LLM to summarize all of the input texts "
        "into a single text based on the instructions provided by the user in their input. "
    ),
    category=ToolCategory.LLM_ANALYSIS,
    reads_chat=True,
)
async def summarize_texts(args: SummarizeTextInput, context: PlanRunContext) -> Text:
    if context.chat is None:
        # just for mypy, shouldn't happen
        return Text(val="")
    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_PLANNER, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_SMART_MODEL)
    result = await llm.do_chat_w_sys_prompt(
        SUMMARIZE_PROMPT_MAIN.format(
            texts="\n***\n".join(Text.get_all_strs(args.texts)),
            chat_context=context.chat.get_gpt_input(),
        ),
        SUMMARIZE_SYS_PROMPT.format(),
    )
    return Text(val=result)


async def topic_filter_helper(
    texts: List[str], topic: str, agent_id: str
) -> List[Tuple[bool, str]]:
    gpt_context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=DEFAULT_SMART_MODEL)
    tasks = []
    for text_str in texts:
        tasks.append(
            llm.do_chat_w_sys_prompt(
                TOPIC_FILTER_MAIN_PROMPT.format(text=text_str, topic=topic),
                TOPIC_FILTER_SYS_PROMPT.format(),
            )
        )
    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)

    counts: Dict[int, int] = defaultdict(int)
    score_tuples = []
    for result in results:
        try:
            rationale, score = result.strip().replace("\n\n", "\n").split("\n")
            score = int(score)
        except ValueError:
            score = 0
            rationale = "No relevance"
        counts[score] += 1
        score_tuples.append((score, rationale))

    if counts[3] > len(texts) * LLM_FILTER_MAX_PERCENT:
        # If there are lot of 3s, only include 3
        cutoff = 3
    elif counts[3] + counts[2] < len(texts) * LLM_FILTER_MAX_PERCENT:
        # If there are hardly any 3 + 2, include 1s
        cutoff = 1
    else:
        # otherwise, includes 2s and 3s
        cutoff = 2

    return [(score >= cutoff, rationale) for score, rationale in score_tuples]


class FilterTextsByTopicInput(ToolArgs):
    topic: str
    texts: List[Text]


@tool(
    description=(
        "This function takes a topic and list of texts and uses an LLM to filter the list of Text to only those"
        " that are relevant to the provided topic. Can be applied to news, earnings, SEC filings, and any"
        " other text. Use this function when you only care about filtering texts for the purposes"
        " display/summarization. This cannot be used with StockAlignedTextGroups, "
        " Use filter_stock_by_topic if you want to filter stocks based on the contents of "
        " associated texts"
    ),
    category=ToolCategory.LLM_ANALYSIS,
)
async def filter_texts_by_topic(
    args: FilterTextsByTopicInput, context: PlanRunContext
) -> List[Text]:
    # not currently returning rationale, but will probably want it
    texts: List[str] = Text.get_all_strs(args.texts)  # type: ignore
    return [
        text
        for text, (is_relevant, _) in zip(
            args.texts, await topic_filter_helper(texts, args.topic, context.agent_id)
        )
        if is_relevant
    ]


class FilterStocksByTopicInput(ToolArgs):
    topic: str
    text_groups: StockAlignedTextGroups


@tool(
    description=(
        "This function takes a StockAlignedTextGroups object which has a mapping between stocks and "
        "some corresponding associated texts"
        " It uses an LLM to filter to only those stocks whose text representation is relevant to the provided topic. "
        "The output of this function is a filtered list of stocks, not a filtered list of "
        "texts. If your goal is filtering texts directly, you should retrieve raw Texts and "
        " use filter_text_by_topic."
        "Again, the output of this function is a list of stocks, not texts!!!"
    ),
    category=ToolCategory.LLM_ANALYSIS,
)
async def filter_stocks_by_topic_aligned(
    args: FilterStocksByTopicInput, context: PlanRunContext
) -> List[int]:
    str_dict: Dict[int, str] = Text.get_all_strs(args.text_groups.val)  # type: ignore
    stocks = list(str_dict.keys())
    texts = list(str_dict.values())
    return [
        stock
        for stock, (is_relevant, _) in zip(
            stocks, await topic_filter_helper(texts, args.topic, context.agent_id)
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
        await stock_identifier_lookup(StockIdentifierLookupInput(stock_name=stock), plan_context)
        for stock in ["Meta", "Apple", "Microsoft"]
    ]  # Convert stock names to identifiers
    print(stock_ids)
    news_developments = await get_all_news_developments_about_companies(
        GetNewsDevelopmentsAboutCompaniesInput(stock_ids=stock_ids, start_date=start_date),  # type: ignore
        plan_context,
    )  # Get news developments for the last month for Meta, Apple, and Microsoft
    print(len(news_developments))  # type: ignore
    summary = await summarize_texts(
        SummarizeTextInput(texts=news_developments, topic="machine learning"), plan_context  # type: ignore
    )  # Summarize the filtered news texts into a single summary
    print(summary)
    news_developments_aligned = await get_stock_aligned_news_developments(
        GetNewsDevelopmentsAboutCompaniesInput(stock_ids=stock_ids, start_date=start_date),  # type: ignore
        plan_context,
    )  # Get news developments for the last month for Meta, Apple, and Microsoft

    filtered_stocks = await filter_stocks_by_topic_aligned(
        FilterStocksByTopicInput(
            topic="Machine Learning", text_groups=news_developments_aligned  # type: ignore
        ),
        plan_context,
    )
    print(filtered_stocks)


if __name__ == "__main__":
    asyncio.run(main())
