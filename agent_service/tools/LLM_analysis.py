import asyncio
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from agent_service.GPT.constants import FILTER_CONCURRENCY, GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import HistoryEntry
from agent_service.io_types.stock import StockAlignedTextGroups, StockID
from agent_service.io_types.text import NewsText, Text
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.dates import DateFromDateStrInput, get_date_from_date_str
from agent_service.tools.news import (  # get_stock_aligned_news_developments,
    GetNewsDevelopmentsAboutCompaniesInput,
    get_all_news_developments_about_companies,
)
from agent_service.tools.stocks import (
    StockIdentifierLookupInput,
    stock_identifier_lookup,
)
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency, identity
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
    template="Summarize the following text(s) based on the needs of the client. Here are the documents, delimited by -----:\n-----\n{texts}\n-----\nHere is the transcript of your interaction with the client, delimited by ----:\n----\n{chat_context}\n----\n{topic_phrase}. Now write your summary",  # noqa: E501
)

TOPIC_FILTER_SYS_PROMPT = Prompt(
    name="TOPIC_FILTER_SYS_PROMPT",
    template="You are a financial analyst checking a text or collection of texts to see if there is anything in the texts that is strongly relevant to the provided topic. On the first line of your output, if you think there is at least some relevance to the topic, please briefly discuss the nature of relevance in no more than 30 words. Just directly highlight any content that is relevant to the topic in your discussion, avoid boilerplate language like `The text discusses` and in fact you absolutely must not refer to `the text`, just talk about the content. If there is absolutely no relevance, you should simply output `No relevance`. Then on the second line, output a number between 0 and 3. 0 indicates no relevance, 1 indicates some relevance, 2 indicates moderate relevance, and 3 should be used when the text is clearly highly relevant to the topic. Most of the texts you will read will not be relevant, and so 0 should be your default.",  # noqa: E501
)

TOPIC_FILTER_MAIN_PROMPT = Prompt(
    name="TOPIC_FILTER_MAIN_PROMPT",
    template="Decide to what degree the following text or texts have information that is relevant to the provided topic. Here is the text or texts, delimited by ---:\n---\n{text}\n---\n. The topic is: {topic}. Write your discussion, followed by your relevant rating between 0 and 3: ",  # noqa: E501
)

ANSWER_QUESTION_SYS_PROMPT = Prompt(
    name="ANSWER_QUESTION_SYS_PROMPT",
    template="You are a financial analyst highly skilled at retrieval of important financial information. You will be provided with a question, and one or more text documents that may contain its answer. Search carefully for the answer, and provide one if you can find it. If you find information that is pertinent to the question but nevertheless does not strictly speaking answer it, you may choose admit that you did not find an answer, but provide the relevant information you did find. If there is no information that is at least somewhat relevant to the question, then simply say that you could not find an answer. For example, if the text provided was simply  `McDonald's has extensive operations in Singapore` and the question was `Does McDonald's have operations in Malaysia?`, you might answer: The information I have direct access to does not indicate whether McDonald's has operations in Malaysia, however it definitely has operations in neighboring Singapore. But if the question was `How did McDonald's revenue in China change last year?`, the information in the text is essentially irrelevant to this question and you need to admit that you have no direct knowledge of the answer to the question. You may use common sense facts to help guide you, but the core of your answer must come from some text provided in your input, you must not answer questions based on extensively on information that is not provided in the input documents. You should limit your answer to no longer than a paragraph of 200 words.",  # noqa: E501
)

ANSWER_QUESTION_MAIN_PROMPT = Prompt(
    name="ANSWER_QUESTION_MAIN_PROMPT",
    template="Answer the following question to the extent that is possible from the information in the text(s) provided, admitting that the information is not there if it is not. Here are the text(s), delimited by '---':\n---\n{texts}\n---\nHere is the question:\n{question}\nNow write your answer: ",  # noqa: E501
)

PROFILE_FILTER_SYS_PROMPT = Prompt(
    name="PROFILE_FILTER_SYS_PROMPT",
    template="You are a financial analyst highly skilled at analyzing documents for insights about companies. You will be given a group of documents which talk about a particular company, and need to decide if the company matches that profile based on the documents you have. Sometimes the profile will contain objective, factual requirements that are easy to verify, please only include stocks where there is strong evidence the condition holds for the company within the documents that have been provided. For example, if you are looking for `companies which produce solar cells`, there must be explicit mention of the company producing such a product somewhere in the documents you have. Other requirements are might be more subjective, for example `companies taking a commanding role in pharmaceutical R&D relative to their peers`, there may not be explicit mention of such nebulous property in the documents, but if you can find at least some significant evidence for it (and no clear counter evidence) such that you can make a case, you should allow the company to pass. If the profile includes multiple requirements, you must be sure that all hold, unless there is an explicit disjunction. For example, if the profiles say `companies that offer both ICE and electric vehicles`, then you must find evidence of both ICE and electric vehicles as products to accept, but if says `companies that offer both either ICE or electric vehicles`, then only one of the two is required (but both is also good). First, output 1 or 2 sentences (no more than 100 words) which justify your choice (state facts from the document(s) and make it clear why they imply the company fits the profile) and then, on a second line, write Yes if you think it does match the profile, or No if it does not. Be conservative, you should say No more often than Yes.",  # noqa: E501
)
PROFILE_FILTER_MAIN_PROMPT = Prompt(
    name="PROFILE_FILTER_MAIN_PROMPT",
    template="Decide whether or not, based on the provided documents related to a company, whether or not it matches the provided profile. Here is the company name: {company_name}. Here are the documents about it, delimited by '---':\n---\n{texts}\n---\nHere is the profile:\n{profile}\nNow discussion your decision, and provide a final answer on the next line:\n",  # noqa: E501
)

TOPIC_PHRASE = "The client has asked for the summary to be focused specifically on the following topic: {topic}. "


# These are to try to force the filter to allow some hits but not too many
LLM_FILTER_MAX_PERCENT = 0.2
LLM_FILTER_MIN_PERCENT = 0.05


class SummarizeTextInput(ToolArgs):
    texts: List[Text]
    topic: Optional[str] = None


@tool(
    description=(
        "This function takes a list of Texts of any kind and uses an LLM to summarize all of the input texts "
        "into a single text based on the instructions provided by the user in their input. "
        "You may also provide a topic if you want the summary to have a very specific focus."
        "If you do this, you should NOT apply filter_texts_by_topic before you run this"
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
    llm = GPT(context=gpt_context, model=GPT4_O)
    texts_str = "\n***\n".join(Text.get_all_strs(args.texts, include_header=True))
    chat_str = context.chat.get_gpt_input()
    topic = args.topic
    if topic:
        topic_str = TOPIC_PHRASE.format(topic=topic)
    else:
        topic_str = ""
    tokenizer = GPTTokenizer(GPT4_O)
    used = tokenizer.get_token_length(
        "\n".join(
            [SUMMARIZE_PROMPT_MAIN.template, SUMMARIZE_SYS_PROMPT.template, chat_str, topic_str]
        )
    )
    texts_str = tokenizer.chop_input_to_allowed_length(texts_str, used)
    result = await llm.do_chat_w_sys_prompt(
        SUMMARIZE_PROMPT_MAIN.format(
            texts=texts_str, chat_context=chat_str, topic_phrase=TOPIC_PHRASE
        ),
        SUMMARIZE_SYS_PROMPT.format(),
    )
    return Text(val=result)


class AnswerQuestionInput(ToolArgs):
    question: str
    texts: List[Text]


@tool(
    description=(
        "This function takes a list of Texts of any kind and searches them for the answer to a question "
        "typically a factual question about a specific stock, e.g. `What countries does Pizza Hut has restaurants in?`"
        " The texts can be any kind of document that might be a potential source of an answer."
        " If the user ask a question, you must try to derive the answer from the text using this function"
        " you cannot just show a text that is likely to have it."
        " When answering questions about a particular stock, you should default to using all text available "
        " for that stock, unless the user particularly asks for a source, or you're 100% sure that "
        " the answer will only be in one particular source."
    ),
    category=ToolCategory.LLM_ANALYSIS,
)
async def answer_question_with_text_data(
    args: AnswerQuestionInput, context: PlanRunContext
) -> Text:
    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_PLANNER, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)
    texts_str = "\n***\n".join(Text.get_all_strs(args.texts, include_header=True))
    tokenizer = GPTTokenizer(GPT4_O)
    used = tokenizer.get_token_length(
        "\n".join([SUMMARIZE_PROMPT_MAIN.template, SUMMARIZE_SYS_PROMPT.template, args.question])
    )
    texts_str = tokenizer.chop_input_to_allowed_length(texts_str, used)
    result = await llm.do_chat_w_sys_prompt(
        ANSWER_QUESTION_MAIN_PROMPT.format(
            texts=texts_str,
            question=args.question,
        ),
        ANSWER_QUESTION_SYS_PROMPT.format(),
    )
    return Text(val=result)


# Topic filter


async def topic_filter_helper(
    texts: List[str], topic: str, agent_id: str
) -> List[Tuple[bool, str]]:
    gpt_context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=GPT4_O)
    tokenizer = GPTTokenizer(GPT4_O)
    used = tokenizer.get_token_length(
        "\n".join([TOPIC_FILTER_MAIN_PROMPT.template, TOPIC_FILTER_SYS_PROMPT.template, topic])
    )
    tasks = []
    for text_str in texts:
        text_str = tokenizer.chop_input_to_allowed_length(text_str, used)
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


class FilterNewsByTopicInput(ToolArgs):
    topic: str
    news_texts: List[NewsText]


@tool(
    description=(
        "This function takes a topic and list of news texts and uses an LLM to filter the texts to only those"
        " that are relevant to the provided topic."
        " Please choose very carefully between this function and filter_stocks_by_profile. This "
        " must NOT be used if your current interest is in filtering stocks/companies, it must only be used when "
        " your immediate goal is filtering news for direct presentation to the user. If you are summarizing news "
        " (or any other Texts) it is better to use that the summarize_text tool directly with a topic argument"
        " For example, if your client asks: "
        " Give me a list of news about Microsoft related to AI"
        " You could apply this filtering function on news developments for Microsoft"
        " Again, both input and output of this function are lists of news texts with no connection to stocks"
        " You must not use this function to filter stocks."
    ),
    category=ToolCategory.LLM_ANALYSIS,
)
async def filter_news_by_topic(
    args: FilterNewsByTopicInput, context: PlanRunContext
) -> List[NewsText]:
    # not currently returning rationale, but will probably want it
    texts: List[str] = Text.get_all_strs(args.news_texts, include_header=True)  # type: ignore
    return [
        text.with_history_entry(
            HistoryEntry(explanation=reason, title=f"Connection to {args.topic}")
        )
        for text, (is_relevant, reason) in zip(
            args.news_texts, await topic_filter_helper(texts, args.topic, context.agent_id)
        )
        if is_relevant
    ]


# Profile filter


async def profile_filter_helper(
    stocks: List[StockID], texts: List[str], profile: str, agent_id: str
) -> List[Tuple[bool, str]]:
    gpt_context = create_gpt_context(GptJobType.AGENT_PLANNER, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=GPT4_O)
    tokenizer = GPTTokenizer(GPT4_O)
    used = tokenizer.get_token_length(
        "\n".join(
            [PROFILE_FILTER_MAIN_PROMPT.template, PROFILE_FILTER_SYS_PROMPT.template, profile]
        )
    )
    tasks = []
    for text_str, stock in zip(texts, stocks):
        text_str = tokenizer.chop_input_to_allowed_length(text_str, used)
        if text_str == "":  # no text, skip
            tasks.append(identity(""))
        tasks.append(
            llm.do_chat_w_sys_prompt(
                PROFILE_FILTER_MAIN_PROMPT.format(
                    company_name=stock.company_name, texts=text_str, profile=profile
                ),
                PROFILE_FILTER_SYS_PROMPT.format(),
            )
        )
    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)

    output_tuples = []
    for result in results:
        try:
            rationale, answer = result.strip().replace("\n\n", "\n").split("\n")
            is_match = answer.lower().startswith("yes")
        except ValueError:
            is_match = False
            rationale = "No match"
        output_tuples.append((is_match, rationale))

    return output_tuples


class FilterStocksByProfileMatch(ToolArgs):
    profile: str
    stocks: List[StockID]
    texts: List[Text]


@tool(
    description=(
        "This function takes a list of stocks and a lists of texts about those stocks"
        " It uses an LLM to filter to only those stocks whose text representation indicates that the stock "
        " matches the provided profile"
        " The profile string must specify the exact property the desired companies have "
        " For example, the profile might be `companies which operate in Spain` or "
        "`companies which produce wingnuts used in Boeing airplanes` "
        " If the client just expresses interest in companies related to a topic X in some way, the profile may also be "
        " `companies with recent Y that mention X` (where Y is a text type and X is the topic)"
        " The profile must contain all information to understand why kind of companies we are looking for "
        " without any other context, do not include anaphoric language, do not say things `stocks like those` "
        " because we will not be able to interpret what `those` means without context"
        " If you are filtering stocks, you will use this function, do not filter using filter_news_by_topic!"
        " For example, if the client asked for a list of stocks which had news about product releases, you "
        " would call this function with the profile `companies with product release news`, you do not need to "
        " filter to press release news first! "
        " The text inputs to this function must be documents specifically about the stocks provided, "
        " the function used to acquire the data must take the same list of stock_ids as arguments. Do not pass "
        " the output of `get_news_articles_for_topics`, it does not have information about stocks."
        " The text input to this function should be all the text data about the company that could reasonably "
        " indicate whether or not the profile matches"
        " The output of this function is a filtered list of stocks, not texts."
        " Never use this function to answer a question about a single stock, use the answer question tool."
    ),
    category=ToolCategory.LLM_ANALYSIS,
)
async def filter_stocks_by_profile_match(
    args: FilterStocksByProfileMatch, context: PlanRunContext
) -> List[StockID]:
    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(args.stocks, args.texts)
    str_dict: Dict[StockID, str] = Text.get_all_strs(aligned_text_groups.val, include_header=True)  # type: ignore
    stocks = list(str_dict.keys())
    texts = list(str_dict.values())
    stock_reason_map = {
        stock: reason
        for stock, (is_relevant, reason) in zip(
            stocks, await profile_filter_helper(stocks, texts, args.profile, context.agent_id)
        )
        if is_relevant
    }
    filtered_stocks = [
        stock.with_history_entry(
            HistoryEntry(
                explanation=stock_reason_map[stock], title=f"Connection to '{args.profile}'"
            )
        )
        for stock in aligned_text_groups.val.keys()
        if stock in stock_reason_map
    ]
    return filtered_stocks


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
        skip_db_commit=True,
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

    filtered_stocks = await filter_stocks_by_profile_match(
        FilterStocksByProfileMatch(
            profile="company involved in Machine Learning", texts=news_developments, stocks=stock_ids  # type: ignore
        ),
        plan_context,
    )
    print(filtered_stocks)


if __name__ == "__main__":
    asyncio.run(main())
