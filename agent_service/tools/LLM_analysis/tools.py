import asyncio
import datetime
import json
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union, cast

from agent_service.GPT.constants import FILTER_CONCURRENCY, GPT4_O, GPT4_O_MINI, SONNET
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import Citation, HistoryEntry, Score
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.text import (
    NewsText,
    StockText,
    Text,
    TextCitation,
    TextGroup,
    TopicProfiles,
)
from agent_service.planner.errors import EmptyInputError, EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.dates import DateFromDateStrInput, get_date_from_date_str
from agent_service.tools.LLM_analysis.constants import (
    DEFAULT_LLM,
    LLM_FILTER_MAX_PERCENT,
    MAX_CITATION_TRIES,
    NO_CITATIONS_DIFF,
    NO_SUMMARY,
    NO_SUMMARY_FOR_STOCK,
    RUBRIC_DELIMITER,
    SCORE_MAPPING,
    SCORE_OUTPUT_DELIMITER,
    UPDATE_PLAN_DELIMITER,
)
from agent_service.tools.LLM_analysis.prompts import (
    ANSWER_QUESTION_DESCRIPTION,
    ANSWER_QUESTION_MAIN_PROMPT,
    ANSWER_QUESTION_SYS_PROMPT,
    COMPARISON_DESCRIPTION,
    COMPARISON_MAIN_PROMPT,
    COMPARISON_SYS_PROMPT,
    COMPARISON_UPDATE_INSTRUCTIONS,
    COMPLEX_PROFILE_FILTER_SYS_PROMPT,
    EXTRA_DATA_PHRASE,
    FILTER_BY_PROFILE_DESCRIPTION,
    FILTER_BY_TOPIC_DESCRIPTION,
    PER_STOCK_SUMMARIZE_DESCRIPTION,
    PROFILE_ADD_DIFF_MAIN_PROMPT,
    PROFILE_ADD_DIFF_SYS_PROMPT,
    PROFILE_FILTER_MAIN_PROMPT,
    PROFILE_REMOVE_DIFF_MAIN_PROMPT,
    PROFILE_REMOVE_DIFF_SYS_PROMPT,
    PROFILE_RUBRIC_GENERATION_MAIN_OBJ,
    PROFILE_RUBRIC_GENERATION_SYS_OBJ,
    RUBRIC_EVALUATION_MAIN_OBJ,
    RUBRIC_EVALUATION_SYS_OBJ,
    SIMPLE_PROFILE_FILTER_SYS_PROMPT,
    SUMMARIZE_DESCRIPTION,
    SUMMARIZE_MAIN_PROMPT,
    SUMMARIZE_SYS_PROMPT,
    SUMMARIZE_UPDATE_INSTRUCTIONS,
    TOPIC_FILTER_MAIN_PROMPT,
    TOPIC_FILTER_SYS_PROMPT,
    TOPIC_PHRASE,
    UPDATE_SUMMARIZE_MAIN_PROMPT,
    UPDATE_SUMMARIZE_SYS_PROMPT,
)
from agent_service.tools.LLM_analysis.utils import extract_citations_from_gpt_output
from agent_service.tools.news import (
    GetNewsDevelopmentsAboutCompaniesInput,
    get_all_news_developments_about_companies,
)
from agent_service.tools.stocks import (
    StockIdentifierLookupInput,
    stock_identifier_lookup,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.string_utils import clean_to_json_if_needed
from agent_service.utils.tool_diff import (
    add_old_history,
    get_prev_run_info,
    get_stock_text_lookup,
    get_text_diff,
)


# Helper functions
# to be depreciated
def split_text_and_citation_ids(GPT_ouput: str) -> Tuple[str, List[int]]:
    logger = get_prefect_logger(__name__)
    lines = GPT_ouput.replace("\n\n", "\n").split("\n")
    try:
        citation_ids = json.loads(clean_to_json_if_needed(lines[-1]))
    except json.JSONDecodeError as e:
        logger.warning(
            f"Got error `{e}` when loading `{lines[-1]}` for citations, no citations included"
        )
        citation_ids = []
    main_text = "\n".join(lines[:-1])
    return main_text, citation_ids


class SummarizeTextInput(ToolArgs):
    texts: List[Text]
    topic: Optional[str] = None


async def _initial_summarize_helper(
    args: SummarizeTextInput, context: PlanRunContext, llm: GPT
) -> Tuple[str, List[TextCitation]]:
    logger = get_prefect_logger(__name__)
    text_group = TextGroup(val=args.texts)
    texts_str: str = await Text.get_all_strs(text_group, include_header=True, text_group_numbering=True)  # type: ignore
    if context.chat:
        chat_str = context.chat.get_gpt_input()
    else:
        chat_str = ""
    topic = args.topic
    if topic:
        topic_str = TOPIC_PHRASE.format(topic=topic)
    else:
        topic_str = ""
    texts_str = GPTTokenizer(DEFAULT_LLM).do_truncation_if_needed(
        texts_str,
        [SUMMARIZE_MAIN_PROMPT.template, SUMMARIZE_SYS_PROMPT.template, chat_str, topic_str],
    )
    if not texts_str:
        raise EmptyInputError("Input text(s) are empty")
    main_prompt = SUMMARIZE_MAIN_PROMPT.format(
        texts=texts_str,
        chat_context=chat_str,
        topic_phrase=topic_str,
        today=(
            context.as_of_date.date().isoformat()
            if context.as_of_date
            else datetime.date.today().isoformat()
        ),
    )
    result = await llm.do_chat_w_sys_prompt(
        main_prompt,
        SUMMARIZE_SYS_PROMPT.format(),
    )
    text, citations = await extract_citations_from_gpt_output(result, text_group, context)
    tries = 0
    while citations is None and tries < MAX_CITATION_TRIES:  # failed to load citations, retry
        logger.warning(f"Retrying after no citations after  {result}")
        result = await llm.do_chat_w_sys_prompt(
            main_prompt, SUMMARIZE_SYS_PROMPT.format(), no_cache=True, temperature=0.1 * (tries + 1)
        )
        text, citations = await extract_citations_from_gpt_output(result, text_group, context)
        tries += 1

    return text or result, citations or []


async def _update_summarize_helper(
    args: SummarizeTextInput,
    context: PlanRunContext,
    llm: GPT,
    new_texts: List[Text],
    old_texts: List[Text],
    old_summary: str,
) -> Tuple[str, List[TextCitation]]:
    logger = get_prefect_logger(__name__)
    new_text_group = TextGroup(val=new_texts)
    new_texts_str: str = await Text.get_all_strs(
        new_text_group, include_header=True, text_group_numbering=True
    )  # type: ignore
    old_texts_group = TextGroup(val=old_texts, offset=len(new_texts))
    old_texts_str: str = await Text.get_all_strs(
        old_texts_group, include_header=True, text_group_numbering=True
    )  # type: ignore

    if context.chat:
        chat_str = context.chat.get_gpt_input()
    else:
        chat_str = ""
    topic = args.topic
    if topic:
        topic_str = TOPIC_PHRASE.format(topic=topic)
    else:
        topic_str = ""

    new_texts_str = GPTTokenizer(DEFAULT_LLM).do_truncation_if_needed(
        new_texts_str,
        [
            UPDATE_SUMMARIZE_MAIN_PROMPT.template,
            UPDATE_SUMMARIZE_SYS_PROMPT.template,
            chat_str,
            topic_str,
            old_texts_str,
        ],
    )

    if not new_texts_str and not old_texts_str:
        raise EmptyInputError("Input text(s) are empty")

    main_prompt = UPDATE_SUMMARIZE_MAIN_PROMPT.format(
        new_texts=new_texts_str,
        old_texts=old_texts_str,
        old_summary=old_summary,
        chat_context=chat_str,
        topic_phrase=topic_str,
        today=(
            context.as_of_date.date().isoformat()
            if context.as_of_date
            else datetime.date.today().isoformat()
        ),
    )

    result = await llm.do_chat_w_sys_prompt(
        main_prompt,
        UPDATE_SUMMARIZE_SYS_PROMPT.format(),
    )

    result = result.split(UPDATE_PLAN_DELIMITER)[-1]  # strip planning section

    combined_text_group = TextGroup.join(new_text_group, old_texts_group)

    text, citations = await extract_citations_from_gpt_output(result, combined_text_group, context)
    tries = 0
    while citations is None and tries < MAX_CITATION_TRIES:  # missing all citations, do a retry
        logger.warning(f"Retrying after no citations for {result}")
        result = await llm.do_chat_w_sys_prompt(
            main_prompt,
            UPDATE_SUMMARIZE_SYS_PROMPT.format(plan_delimiter=UPDATE_PLAN_DELIMITER),
            no_cache=True,
            temperature=0.1 * (tries + 1),
        )
        result = result.split(UPDATE_PLAN_DELIMITER)[-1]  # strip planning section

        text, citations = await extract_citations_from_gpt_output(
            result, combined_text_group, context
        )

        tries += 1

    return text or result, citations or []


@tool(
    description=SUMMARIZE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
    reads_chat=True,
    update_instructions=SUMMARIZE_UPDATE_INSTRUCTIONS,
)
async def summarize_texts(args: SummarizeTextInput, context: PlanRunContext) -> Text:

    logger = get_prefect_logger(__name__)
    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_LLM)

    if len(args.texts) == 0:
        raise EmptyInputError("Cannot summarize when no texts provided")

    text = None
    citations = None

    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "summarize_texts")
        if prev_run_info is not None:
            prev_args = SummarizeTextInput.model_validate_json(prev_run_info.inputs_str)
            prev_texts: List[Text] = prev_args.texts
            prev_output: Text = prev_run_info.output  # type:ignore
            curr_input_texts = set(args.texts)
            new_texts = list(curr_input_texts - set(prev_texts))
            all_old_citations = cast(List[TextCitation], prev_output.history[0].citations)
            remaining_citations = [
                citation
                for citation in all_old_citations
                if citation.source_text in curr_input_texts
            ]
            if new_texts or (
                remaining_citations and len(all_old_citations) > len(remaining_citations)
            ):
                old_texts = [citation.source_text for citation in remaining_citations]
                text, citations = await _update_summarize_helper(
                    args, context, llm, new_texts, old_texts, prev_output.val
                )
            else:
                if remaining_citations:  # output old summary
                    text = prev_output.val
                    citations = remaining_citations
                else:  # unless there's nothing left to cite
                    text = NO_SUMMARY
                    citations = []

    except Exception as e:
        logger.warning(
            f"Failed attempt to update from previous iteration due to {e}, from scratch fallback"
        )

    if text is None:
        text, citations = await _initial_summarize_helper(args, context, llm)

    summary: Text = Text(val=text)
    summary = summary.inject_history_entry(
        HistoryEntry(title="Summary", citations=citations)  # type:ignore
    )
    return summary


class PerStockSummarizeTextInput(ToolArgs):
    stocks: List[StockID]
    texts: List[StockText]
    topic: str


@tool(
    description=PER_STOCK_SUMMARIZE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
    reads_chat=True,
)
async def per_stock_summarize_texts(
    args: PerStockSummarizeTextInput, context: PlanRunContext
) -> List[StockID]:

    logger = get_prefect_logger(__name__)
    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_LLM)

    text_dict: Dict[StockID, List[StockText]] = {stock: [] for stock in args.stocks}
    for text in args.texts:
        try:
            if text.stock_id in text_dict:
                text_dict[text.stock_id].append(text)
        except AttributeError:
            logger.warning("Non-StockText passed to per stock summarize")

    prev_run_dict: Dict[StockID, Tuple[List[StockText], List[TextCitation], bool, str]] = (
        defaultdict()
    )

    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "per_stock_summarize_texts")
        if prev_run_info is not None:
            prev_args = PerStockSummarizeTextInput.model_validate_json(prev_run_info.inputs_str)
            prev_texts: List[StockText] = prev_args.texts
            prev_output: List[StockID] = prev_run_info.output  # type:ignore

            old_text_dict: Dict[StockID, List[StockText]] = defaultdict(list)
            for text in prev_texts:
                try:
                    if text.stock_id in text_dict:
                        old_text_dict[text.stock_id].append(text)
                except AttributeError:
                    logger.warning("Non-StockText passed to per stock summarize")

            for stock in prev_output:
                curr_input_texts = set(text_dict.get(stock, []))
                old_input_texts = set(old_text_dict.get(stock, []))
                new_texts = list(curr_input_texts - old_input_texts)
                old_summary = cast(str, stock.history[-1].explanation)
                all_old_citations = cast(List[TextCitation], stock.history[-1].citations)
                remaining_citations = [
                    citation
                    for citation in all_old_citations
                    if citation.source_text in curr_input_texts
                ]
                prev_run_dict[stock] = (
                    new_texts,
                    remaining_citations,
                    len(all_old_citations) > len(remaining_citations),
                    old_summary,
                )

    except Exception as e:
        logger.warning(
            f"Failed attempt to update from previous iteration due to {e}, from scratch fallback"
        )

    tasks = []
    summary_count = 0

    for stock in args.stocks:
        if stock in text_dict:
            if stock in prev_run_dict:
                new_texts, remaining_citations, has_lost_citations, old_summary = prev_run_dict[
                    stock
                ]
                old_texts = [citation.source_text for citation in remaining_citations]
                if new_texts or (remaining_citations and has_lost_citations):
                    tasks.append(
                        _update_summarize_helper(
                            SummarizeTextInput(texts=new_texts, topic=args.topic),  # type: ignore
                            context,
                            llm,
                            new_texts,  # type: ignore
                            old_texts,
                            old_summary,
                        )
                    )
                else:
                    if remaining_citations:  # no new texts, just use old summary
                        tasks.append(identity((old_summary, remaining_citations)))
                    else:  # unless there is nothing to cite, at which point use default
                        tasks.append(identity((NO_SUMMARY_FOR_STOCK, [])))

            else:
                tasks.append(
                    _initial_summarize_helper(
                        SummarizeTextInput(texts=text_dict[stock], topic=args.topic),  # type: ignore
                        context,
                        llm,
                    )
                )
            summary_count += 1
        else:
            tasks.append(identity(None))

    await tool_log(
        f"Writing texts for {summary_count} stocks, topic: {args.topic}", context=context
    )

    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)
    output = []

    for stock, (summary, citations) in zip(args.stocks, results):
        stock = stock.inject_history_entry(
            HistoryEntry(
                explanation=summary,
                title=args.topic,
                citations=citations,  # type:ignore
                task_id=context.task_id,
            )
        )
        output.append(stock)

    return output


class CompareTextInput(ToolArgs):
    group1: List[Text]
    group1_label: str
    group2: List[Text]
    group2_label: str
    extra_data: Optional[List[Text]] = None
    extra_data_label: Optional[str] = None


@tool(
    description=COMPARISON_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
    reads_chat=True,
    update_instructions=COMPARISON_UPDATE_INSTRUCTIONS,
)
async def compare_texts(args: CompareTextInput, context: PlanRunContext) -> Text:
    if context.chat is None:
        # just for mypy, shouldn't happen
        return Text(val="")
    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_LLM)
    text_group1 = TextGroup(val=args.group1)
    group1_str: str = await Text.get_all_strs(
        text_group1, include_header=True, text_group_numbering=True
    )  # type: ignore
    text_group2 = TextGroup(val=args.group2, offset=len(args.group1))
    group2_str: str = await Text.get_all_strs(
        text_group2, include_header=True, text_group_numbering=True
    )  # type: ignore
    if args.extra_data is not None and args.extra_data_label is not None:
        extra_group = TextGroup(val=args.extra_data)
        extra_data_str = EXTRA_DATA_PHRASE.format(
            extra_data=await Text.get_all_strs(extra_group), label=args.extra_data_label  # type: ignore
        )
    else:
        extra_data_str = ""
    chat_str = context.chat.get_gpt_input()
    # FIXME using GPT4 tokenizer/limits for SONNET
    group1_str, group2_str = GPTTokenizer(DEFAULT_LLM).do_multi_truncation_if_needed(
        [group1_str, group2_str],
        [
            COMPARISON_MAIN_PROMPT.template,
            COMPARISON_SYS_PROMPT.template,
            chat_str,
            extra_data_str,
        ],
    )
    result = await llm.do_chat_w_sys_prompt(
        COMPARISON_MAIN_PROMPT.format(
            group1=group1_str,
            group2=group2_str,
            group1_label=args.group1_label,
            group2_label=args.group2_label,
            extra_data=extra_data_str,
            chat_context=chat_str,
            today=(
                context.as_of_date.date().isoformat()
                if context.as_of_date
                else datetime.date.today().isoformat()
            ),
        ),
        COMPARISON_SYS_PROMPT.format(),
    )

    merged_group = TextGroup.join(text_group1, text_group2)
    main_text, citations = await extract_citations_from_gpt_output(result, merged_group, context)
    comparison: Text = Text(val=main_text)
    comparison = comparison.inject_history_entry(
        HistoryEntry(title="Text Comparison", citations=citations)  # type:ignore
    )
    return comparison


class AnswerQuestionInput(ToolArgs):
    question: str
    texts: List[Text]


@tool(
    description=ANSWER_QUESTION_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
)
async def answer_question_with_text_data(
    args: AnswerQuestionInput, context: PlanRunContext
) -> Text:
    if len(args.texts) == 0:
        raise EmptyInputError(message="Cannot answer question with an empty list of texts")
    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_LLM)
    text_group = TextGroup(val=args.texts)
    texts_str: str = await Text.get_all_strs(text_group, include_header=True, text_group_numbering=True)  # type: ignore
    texts_str = GPTTokenizer(DEFAULT_LLM).do_truncation_if_needed(
        texts_str,
        [ANSWER_QUESTION_MAIN_PROMPT.template, ANSWER_QUESTION_SYS_PROMPT.template, args.question],
    )

    result = await llm.do_chat_w_sys_prompt(
        ANSWER_QUESTION_MAIN_PROMPT.format(
            texts=texts_str,
            question=args.question,
            today=(
                context.as_of_date.date().isoformat()
                if context.as_of_date
                else datetime.date.today().isoformat()
            ),
        ),
        ANSWER_QUESTION_SYS_PROMPT.format(),
    )

    text, citations = await extract_citations_from_gpt_output(result, text_group, context)
    answer: Text = Text(val=text)
    answer = answer.inject_history_entry(
        HistoryEntry(title="Summary", citations=citations)  # type:ignore
    )
    return answer


# Topic filter


async def topic_filter_helper(
    texts: List[str], topic: str, agent_id: str
) -> List[Tuple[bool, str]]:
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
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
    description=FILTER_BY_TOPIC_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
)
async def filter_news_by_topic(
    args: FilterNewsByTopicInput, context: PlanRunContext
) -> List[NewsText]:
    if len(args.news_texts) == 0:
        raise EmptyInputError("Cannot filter empty list of texts")
    # not currently returning rationale, but will probably want it
    texts: List[str] = await Text.get_all_strs(args.news_texts, include_header=True)  # type: ignore
    return [
        text.inject_history_entry(
            HistoryEntry(explanation=reason, title=f"Connection to {args.topic}")
        )
        for text, (is_relevant, reason) in zip(
            args.news_texts, await topic_filter_helper(texts, args.topic, context.agent_id)
        )
        if is_relevant
    ]


# Profile filter


async def profile_filter_helper(
    aligned_text_groups: StockAlignedTextGroups,
    str_lookup: Dict[StockID, str],
    profile: str,
    is_using_complex_profile: bool,
    llm: GPT,
    context: PlanRunContext,
    topic: str = "",
    do_citations: bool = True,
    stock_whitelist: Optional[Set[StockID]] = None,
) -> List[Tuple[bool, str, List[Citation]]]:
    tokenizer = GPTTokenizer(GPT4_O)
    used = tokenizer.get_token_length(
        "\n".join(
            [
                PROFILE_FILTER_MAIN_PROMPT.template,
                SIMPLE_PROFILE_FILTER_SYS_PROMPT.template,
                profile,
            ]
        )
    )

    tasks = []
    for stock in aligned_text_groups.val:
        text_str = str_lookup[stock]
        text_str = tokenizer.chop_input_to_allowed_length(text_str, used)
        if text_str == "" or (stock_whitelist and stock in stock_whitelist):  # no text, skip
            tasks.append(identity(""))

        elif is_using_complex_profile:
            tasks.append(
                llm.do_chat_w_sys_prompt(
                    PROFILE_FILTER_MAIN_PROMPT.format(
                        company_name=stock.company_name,
                        texts=text_str,
                        profile=profile,
                        today=(
                            context.as_of_date.date().isoformat()
                            if context.as_of_date
                            else datetime.date.today().isoformat()
                        ),
                    ),
                    COMPLEX_PROFILE_FILTER_SYS_PROMPT.format(topic_name=topic),
                )
            )
        else:
            tasks.append(
                llm.do_chat_w_sys_prompt(
                    PROFILE_FILTER_MAIN_PROMPT.format(
                        company_name=stock.company_name,
                        texts=text_str,
                        profile=profile,
                        today=(
                            context.as_of_date.date().isoformat()
                            if context.as_of_date
                            else datetime.date.today().isoformat()
                        ),
                    ),
                    SIMPLE_PROFILE_FILTER_SYS_PROMPT.format(),
                )
            )

    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)

    output_tuples: List[Tuple[bool, str, List[Citation]]] = []
    for result, text_group in zip(results, aligned_text_groups.val.values()):
        try:
            rationale, answer, citation_anchor_map = (
                result.strip().replace("\n\n", "\n").split("\n")
            )
            is_match = answer.lower().startswith("yes")
            if is_match and do_citations:
                rationale, citations = await extract_citations_from_gpt_output(
                    "\n".join([rationale, citation_anchor_map]), text_group, context
                )
            else:
                citations = []
        except ValueError:
            is_match = False
            rationale = "No match"
            citations = []
        output_tuples.append((is_match, rationale, citations))  # type:ignore

    return output_tuples


async def get_profile_rubric(profile: str, agent_id: str) -> Dict[int, str]:
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=SONNET)
    result = await llm.do_chat_w_sys_prompt(
        main_prompt=PROFILE_RUBRIC_GENERATION_MAIN_OBJ.format(profile=profile),
        sys_prompt=PROFILE_RUBRIC_GENERATION_SYS_OBJ.format(),
        max_tokens=2000,
    )
    # We don't care about the justification at the top, just meant to keep
    # GPT whipped
    rubric_dict = {k: "" for k in range(1, 6)}
    rubric_str = result.split(RUBRIC_DELIMITER)[1].strip()

    # We output the rubric from the LLM by starting the description of
    # each level with "Level" as follows:
    #
    #   Level 1: Description
    #   Level 2: Description
    #   ...
    #
    # Splitting by "Level " then helps to grab all the level descriptions
    # along with the level number associated with it, we then apply
    # some indexing to grab the Description component and strip trailing
    # line breaks
    for entry in rubric_str.split("Level "):
        # Check if empty string
        if entry:
            if isinstance(rubric_dict.get(int(entry[0]), None), str):
                rubric_dict[int(entry[0])] = entry[2:].strip()
    return rubric_dict


async def filtered_stocks_score_assignment(
    stocks: List[StockID],
    rubric_dict: Dict[int, str],
    stock_reason_map: Dict[StockID, Tuple[str, List[Citation]]],
    profile: str,
    context: PlanRunContext,
) -> List[StockID]:
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=SONNET)
    tasks = []

    rubric_str_list = [f"Level {k}: {v}" for k, v in rubric_dict.items()]

    for stock in stocks:
        tasks.append(
            llm.do_chat_w_sys_prompt(
                main_prompt=RUBRIC_EVALUATION_MAIN_OBJ.format(
                    company_name=stock.company_name,
                    reason=stock_reason_map[stock][0],
                ),
                sys_prompt=RUBRIC_EVALUATION_SYS_OBJ.format(
                    rubric_str="\n".join(rubric_str_list),
                ),
            )
        )
    scores = await gather_with_concurrency(tasks, 20)

    non_zero_scoring_stocks = []
    for stock, score in zip(stocks, scores):
        level_score, _ = score.split(SCORE_OUTPUT_DELIMITER)
        # if level_score != "0":
        non_zero_scoring_stocks.append(
            stock.inject_history_entry(
                HistoryEntry(
                    explanation=stock_reason_map[stock][0],
                    title=f"Connection to '{profile}'",
                    score=Score(val=SCORE_MAPPING[level_score]),
                    citations=stock_reason_map[stock][1],
                    task_id=context.task_id,
                )
            )
        )
    return non_zero_scoring_stocks


async def profile_filter_added_diff_info(
    added_stocks: List[StockID],
    profile_str: str,
    stock_text_diff: Dict[StockID, List[StockText]],
    agent_id: str,
) -> Dict[StockID, str]:
    # For each stock that has been added in this run of the profile filter, try to generate a useful explanation
    # for why based on text differences
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=SONNET)

    # might later do citations for this but not bothering now
    str_lookup: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        {stock: stock_text_diff[stock] for stock in added_stocks},
        include_header=True,
        text_group_numbering=False,
    )

    tasks = []
    for stock in added_stocks:
        tasks.append(
            llm.do_chat_w_sys_prompt(
                PROFILE_ADD_DIFF_MAIN_PROMPT.format(
                    company_name=stock.company_name,
                    profiles=profile_str,
                    new_documents=str_lookup[stock],
                ),
                PROFILE_ADD_DIFF_SYS_PROMPT.format(),
            )
        )

    results = await gather_with_concurrency(tasks)
    return {
        stock: explanation
        for stock, explanation in zip(added_stocks, results)
        if not explanation.startswith("No clear evidence")
    }


async def profile_filter_removed_diff_info(
    removed_stocks: List[StockID],
    profile_str: str,
    stock_text_diff: Dict[StockID, List[StockText]],
    agent_id: str,
) -> Dict[StockID, str]:
    # For each stock that has been removed in this run of the profile filter, try to generate a useful explanation
    # for why based on text differences
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=SONNET)

    # might later do citations for this but not bothering now
    str_lookup: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        {stock: stock_text_diff[stock] for stock in removed_stocks},
        include_header=True,
        text_group_numbering=False,
    )

    tasks = []
    for stock in removed_stocks:
        tasks.append(
            llm.do_chat_w_sys_prompt(
                PROFILE_REMOVE_DIFF_MAIN_PROMPT.format(
                    company_name=stock.company_name,
                    profiles=profile_str,
                    new_documents=str_lookup[stock],
                ),
                PROFILE_REMOVE_DIFF_SYS_PROMPT.format(),
            )
        )

    results = await gather_with_concurrency(tasks)
    return {
        stock: explanation
        for stock, explanation in zip(removed_stocks, results)
        if not explanation.startswith("No clear evidence")
    }


class FilterStocksByProfileMatch(ToolArgs):
    stocks: List[StockID]
    texts: List[StockText]
    profile: Union[str, TopicProfiles]


@tool(
    description=FILTER_BY_PROFILE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
)
async def filter_stocks_by_profile_match(
    args: FilterStocksByProfileMatch, context: PlanRunContext
) -> List[StockID]:
    if len(args.stocks) == 0:
        raise EmptyInputError("Cannot filter empty list of stocks")
    if len(args.texts) == 0:
        raise EmptyInputError("Cannot filter stocks with empty list of texts")

    logger = get_prefect_logger(__name__)
    if context.task_id is None:
        return []  # for mypy

    prev_run_info = None
    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "filter_stocks_by_profile_match")
        if prev_run_info is not None:
            prev_args = FilterStocksByProfileMatch.model_validate_json(prev_run_info.inputs_str)
            prev_output: List[StockID] = prev_run_info.output  # type:ignore

            # start by finding the differences in the input texts for the two runs
            prev_stock_text_lookup = get_stock_text_lookup(prev_args.texts)
            current_stock_text_lookup = get_stock_text_lookup(args.texts)
            diff_text_stocks = []
            same_text_stocks = []
            stock_text_diff = {}
            for stock in args.stocks:
                text_diff = get_text_diff(
                    current_stock_text_lookup.get(stock, []), prev_stock_text_lookup.get(stock, [])
                )
                if text_diff:
                    diff_text_stocks.append(stock)
                    stock_text_diff[stock] = text_diff
                else:
                    same_text_stocks.append(stock)

            # we are only going to do main pass for stocks that have some text difference
            # relative to previous run
            args.stocks = diff_text_stocks

    except Exception as e:
        logger.warning(f"Error doing text diff from previous run: {e}")

    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(args.stocks, args.texts)
    str_lookup: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val, include_header=True, text_group_numbering=True
    )

    profile_str: str = ""
    if isinstance(args.profile, TopicProfiles):
        is_using_complex_profile = True
        profile_str = await Text.get_all_strs(  # type: ignore
            args.profile, include_header=False, text_group_numbering=False
        )
        await tool_log(f"Using advanced profile with topic: {args.profile.topic}", context=context)
    elif isinstance(args.profile, str):
        is_using_complex_profile = False
        profile_str = args.profile
        await tool_log(f"Using simple profile: {profile_str}", context=context)
    else:
        raise ValueError(
            "profile must be either a string or a TopicProfiles object "
            "in filter_stocks_by_profile_match function!"
        )
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    cheap_llm = GPT(context=gpt_context, model=GPT4_O_MINI)
    stock_whitelist: Set[StockID] = set()
    await tool_log(
        f"Starting filtering with {len(aligned_text_groups.val.keys())} stocks", context=context
    )
    for stock, (is_relevant, reason, citations) in zip(
        aligned_text_groups.val.keys(),
        await profile_filter_helper(
            aligned_text_groups,
            str_lookup,
            profile_str,
            is_using_complex_profile,
            llm=cheap_llm,
            context=context,
            do_citations=False,
        ),
    ):
        if is_relevant:
            stock_whitelist.add(stock)

    await tool_log(
        f"First round of filtering complete. {len(stock_whitelist)} stocks remaining.",
        context=context,
    )
    llm = GPT(context=gpt_context, model=SONNET)
    stock_reason_map: Dict[StockID, Tuple[str, List[Citation]]] = {
        stock: (reason, citations)
        for stock, (is_relevant, reason, citations) in zip(
            aligned_text_groups.val.keys(),
            await profile_filter_helper(
                aligned_text_groups,
                str_lookup,
                profile_str,
                is_using_complex_profile,
                llm=llm,
                context=context,
                do_citations=True,
                stock_whitelist=stock_whitelist,
            ),
        )
        if is_relevant
    }

    filtered_stocks = [
        stock for stock in aligned_text_groups.val.keys() if stock in stock_reason_map
    ]

    await tool_log(
        f"Second round of filtering complete. {len(filtered_stocks)} stocks remaining.",
        context=context,
    )

    # No need for an else since we can guarantee at this point one is not None, appeases linter
    if isinstance(args.profile, TopicProfiles):
        # TODO: Update the rubric to handle the new extensive profile data we have, for now
        # we just pass in the topic which is a short simple string similar to a profile string
        profile_data_for_rubric = args.profile.topic
    elif isinstance(args.profile, str):
        profile_data_for_rubric = args.profile

    rubric_dict = await get_profile_rubric(profile_data_for_rubric, context.agent_id)
    # Assigns scores inplace
    filtered_stocks_with_scores = await filtered_stocks_score_assignment(
        filtered_stocks, rubric_dict, stock_reason_map, profile_data_for_rubric, context
    )

    try:  # since everything associated with diffing is optional, put in try/except
        if prev_run_info is not None:
            if context.diff_info is not None:  # collect info for automatic diff
                prev_input_set = set(prev_args.stocks)
                prev_output_set = set(prev_output)
                added_stocks = []
                for stock in filtered_stocks_with_scores:
                    if stock in prev_input_set and stock not in prev_output_set:
                        # stock included this time, but not previous time
                        added_stocks.append(stock)

                added_diff_info = await profile_filter_added_diff_info(
                    added_stocks, profile_str, stock_text_diff, context.agent_id
                )

                # get rid of output where we didn't get a good diff
                filtered_stocks_with_scores = [
                    stock for stock in filtered_stocks_with_scores if stock in added_diff_info
                ]

                current_input_set = set(args.stocks)
                current_output_set = set(filtered_stocks_with_scores)

                removed_stocks = []

                # identify stocks that didn't make it through this pass of the tool but did last time

                for stock in prev_output:
                    if stock in current_input_set and stock not in current_output_set:
                        removed_stocks.append(stock)

                removed_diff_info = await profile_filter_removed_diff_info(
                    removed_stocks, profile_str, stock_text_diff, context.agent_id
                )

                # don't remove stocks where we couldn't get a good diff explanation
                # as long as we also didn't lose citations

                for old_stock in removed_stocks:
                    if old_stock not in removed_diff_info:
                        for new_stock in current_input_set:
                            if new_stock == stock:
                                break
                        stock_with_old_history = add_old_history(
                            new_stock,
                            old_stock,
                            context.task_id,
                            current_stock_text_lookup[new_stock],
                        )
                        if stock_with_old_history:  # successful added old history
                            filtered_stocks_with_scores.append(stock_with_old_history)
                        else:  # lost citations
                            removed_diff_info[stock] = NO_CITATIONS_DIFF

                context.diff_info[context.task_id] = {
                    "added": added_diff_info,
                    "removed": removed_diff_info,
                }

    except Exception as e:
        logger.warning(f"Error doing text diff from previous run: {e}")

    try:  # we do this part separately because we don't want to accidently lose the same_text_stocks
        if prev_run_info is not None:
            if same_text_stocks:  # add in results where no change in text
                for new_stock in same_text_stocks:
                    found_stock = False
                    for old_stock in prev_output:
                        if old_stock == new_stock:
                            found_stock = True
                            break
                    if found_stock:
                        stock_with_old_history = add_old_history(
                            new_stock, old_stock, context.task_id
                        )
                        if stock_with_old_history:
                            filtered_stocks_with_scores.append(stock_with_old_history)

    except Exception as e:
        logger.warning(f"Error duplicating output for stocks with no text changes: {e}")

    if isinstance(args.profile, TopicProfiles):
        await tool_log(
            f"Filtered {len(filtered_stocks_with_scores)} stocks for profile: {args.profile.topic}",
            context=context,
        )
    elif isinstance(args.profile, str):
        await tool_log(
            f"Filtered {len(filtered_stocks_with_scores)} stocks for profile: {profile_str}",
            context=context,
        )

    if not filtered_stocks_with_scores:
        raise EmptyOutputError(message="Stock profile filter resulted in an empty list of stocks")
    # dedup stocks
    company_names = set()
    dedup_res = []
    for stock in filtered_stocks_with_scores:
        if stock.company_name not in company_names:
            company_names.add(stock.company_name)
            dedup_res.append(stock)
    return dedup_res


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
