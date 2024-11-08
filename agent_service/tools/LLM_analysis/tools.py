import asyncio
import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from agent_service.GPT.constants import (
    FILTER_CONCURRENCY,
    GPT4_O,
    GPT4_O_MINI,
    MAX_TOKENS,
    SONNET,
)
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import Citation, HistoryEntry, dump_io_type
from agent_service.io_types.idea import Idea
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.stock_groups import StockGroup, StockGroups
from agent_service.io_types.text import (
    DEFAULT_TEXT_TYPE,
    NewsText,
    StockText,
    Text,
    TextCitation,
    TextGroup,
    TopicProfiles,
)
from agent_service.planner.errors import (
    BadInputError,
    EmptyInputError,
    EmptyOutputError,
)
from agent_service.tool import (
    TOOL_DEBUG_INFO,
    ToolArgMetadata,
    ToolArgs,
    ToolCategory,
    tool,
)
from agent_service.tools.dates import DateFromDateStrInput, get_date_from_date_str
from agent_service.tools.ideas.utils import ideas_enabled
from agent_service.tools.LLM_analysis.constants import (
    BRAINSTORM_DELIMITER,
    DEFAULT_LLM,
    IDEA,
    LLM_FILTER_MAX_INPUT_PERCENTAGE,
    LLM_FILTER_MAX_PERCENT,
    LLM_FILTER_MIN_TOKENS,
    MAX_CITATION_DROP_RATIO,
    MAX_CITATION_TRIES,
    NO_CITATIONS_DIFF,
    NO_SUMMARY,
    NO_SUMMARY_FOR_STOCK,
    STOCK_TYPE,
    SUMMARIZE_CITATION_INCREASE_LIMIT,
    TARGET_STOCK,
)
from agent_service.tools.LLM_analysis.prompts import (
    ANSWER_QUESTION_DESCRIPTION,
    ANSWER_QUESTION_MAIN_PROMPT_STR_DEFAULT,
    ANSWER_QUESTION_SYS_PROMPT_STR_DEFAULT,
    BRAINSTORM_FINAL_REMINDER,
    BRAINSTORM_REMINDER,
    CITATION_PROMPT,
    CITATION_REMINDER,
    COMPARISON_DESCRIPTION,
    COMPARISON_MAIN_PROMPT_STR_DEFAULT,
    COMPARISON_SYS_PROMPT_STR_DEFAULT,
    COMPARISON_UPDATE_INSTRUCTIONS,
    COMPLEX_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT,
    EXTRA_DATA_PHRASE,
    FILTER_BY_PROFILE_DESCRIPTION,
    FILTER_BY_TOPIC_DESCRIPTION,
    PER_IDEA_SUMMARIZE_DESCRIPTION,
    PER_STOCK_GROUP_SUMMARIZE_DESCRIPTION,
    PER_STOCK_SUMMARIZE_DESCRIPTION,
    PER_SUMMARIZE_BRAINSTORMING,
    PER_SUMMARIZE_INSTRUCTIONS,
    PER_SUMMARIZE_REMINDER,
    PER_TOPIC_FILTER_BY_PROFILE_DESCRIPTION,
    PROFILE_ADD_DIFF_MAIN_PROMPT,
    PROFILE_ADD_DIFF_SYS_PROMPT,
    PROFILE_FILTER_MAIN_PROMPT_STR_DEFAULT,
    PROFILE_OUTPUT_INSTRUCTIONS_DEFAULT,
    PROFILE_REMOVE_DIFF_MAIN_PROMPT,
    PROFILE_REMOVE_DIFF_SYS_PROMPT,
    SIMPLE_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT,
    STOCK_PHRASE_DIVERSITY,
    STOCK_PHRASE_STR_DEFAULT,
    SUMMARIZE_BRAINSTORM_INSTRUCTIONS_STR_DEFAULT,
    SUMMARIZE_DESCRIPTION,
    SUMMARIZE_MAIN_PROMPT_STR_DEFAULT,
    SUMMARIZE_SYS_PROMPT_STR_DEFAULT,
    SUMMARIZE_UPDATE_INSTRUCTIONS,
    TOPIC_FILTER_MAIN_PROMPT,
    TOPIC_FILTER_SYS_PROMPT,
    TOPIC_PHRASE_STR_DEFAULT,
    UPDATE_SUMMARIZE_MAIN_PROMPT,
    UPDATE_SUMMARIZE_SYS_PROMPT,
)
from agent_service.tools.LLM_analysis.utils import (
    classify_stock_text_relevancies_for_profile,
    extract_citations_from_gpt_output,
    get_all_text_citations,
    get_original_cite_count,
    get_second_order_citations,
    is_topical,
)
from agent_service.tools.news import (
    GetNewsDevelopmentsAboutCompaniesInput,
    get_all_news_developments_about_companies,
)
from agent_service.tools.stock_rank_by_text.utils import (
    get_profile_rubric,
    stocks_rubric_score_assignment,
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
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.text_utils import partition_to_smaller_text_sizes
from agent_service.utils.tool_diff import (
    add_old_history,
    get_prev_run_info,
    get_stock_text_lookup,
    get_text_diff,
)


class SummarizeTextInput(ToolArgs):
    texts: List[Text]
    topic: Optional[str] = None
    stock: Optional[StockID] = None

    # prompt arguments (hidden from planner)
    summarize_sys_prompt: str = SUMMARIZE_SYS_PROMPT_STR_DEFAULT
    summarize_main_prompt: str = SUMMARIZE_MAIN_PROMPT_STR_DEFAULT
    summarize_brainstorm_instruction: str = SUMMARIZE_BRAINSTORM_INSTRUCTIONS_STR_DEFAULT
    topic_phrase: str = TOPIC_PHRASE_STR_DEFAULT
    stock_phrase: str = STOCK_PHRASE_STR_DEFAULT

    # tool arguments metadata
    arg_metadata = {
        "summarize_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
        "summarize_main_prompt": ToolArgMetadata(hidden_from_planner=True),
        "summarize_brainstorm_instruction": ToolArgMetadata(hidden_from_planner=True),
        "topic_phrase": ToolArgMetadata(hidden_from_planner=True),
        "stock_phrase": ToolArgMetadata(hidden_from_planner=True),
    }


async def _initial_summarize_helper(
    args: SummarizeTextInput,
    context: PlanRunContext,
    llm: GPT,
    single_summary: bool = True,
    topic_filter: bool = True,
) -> Tuple[str, List[TextCitation]]:
    # create prompts
    summarize_sys_prompt = Prompt(
        name="LLM_SUMMARIZE_SYS_PROMPT",
        template=args.summarize_sys_prompt + CITATION_PROMPT,
    )
    summarize_main_prompt = Prompt(
        name="LLM_SUMMARIZE_MAIN_PROMPT",
        template=args.summarize_main_prompt
        + CITATION_REMINDER
        + " Pay specific attention to the way that the client wants you to output the summary, "
        + "YOU MUST comply with this format."
        + " Now proceed with your summary writing{final_reminder}:\n",
    )
    summarize_brainstorm_instruction = args.summarize_brainstorm_instruction
    topic_phrase = args.topic_phrase
    stock_phrase = args.stock_phrase
    logger = get_prefect_logger(__name__)
    if args.topic and topic_filter:
        texts = await topic_filter_helper(
            args.texts,
            args.topic,
            context.agent_id,
            model_for_filter_to_context=GPT4_O,
        )
        if len(texts) == 0:  # filtered out all relevant texts
            if single_summary:
                await tool_log(log="All texts filtered out before summary", context=context)
            if args.stock:
                return NO_SUMMARY_FOR_STOCK, []
            else:
                return NO_SUMMARY, []
        if single_summary:
            await tool_log(
                log="Applied topic filtering before summarization, "
                f"{len(args.texts) - len(texts)} of {len(args.texts)} text snippets removed",
                context=context,
            )
    else:
        if single_summary:
            await tool_log(log="No topic filtering", context=context)
        texts = args.texts

    text_group = TextGroup(val=texts)
    texts_str: str = await Text.get_all_strs(  # type: ignore
        text_group, include_header=True, text_group_numbering=True, include_symbols=True
    )
    if context.chat:
        chat_str = context.chat.get_gpt_input()
    else:
        chat_str = ""

    if single_summary:
        brainstorm_str = summarize_brainstorm_instruction.format(
            brainstorm_delimiter=BRAINSTORM_DELIMITER
        )
        brainstorm_reminder = BRAINSTORM_REMINDER
        final_reminder = BRAINSTORM_FINAL_REMINDER
    else:
        brainstorm_str = PER_SUMMARIZE_INSTRUCTIONS.format(
            no_summary=NO_SUMMARY
        ) + PER_SUMMARIZE_BRAINSTORMING.format(no_summary=NO_SUMMARY)
        brainstorm_reminder = ""
        final_reminder = PER_SUMMARIZE_REMINDER

    topic = args.topic
    if topic:
        plan_str = (
            get_psql(skip_commit=context.skip_db_commit)
            .get_execution_plan_for_run(context.plan_run_id)
            .get_formatted_plan(numbered=True)
        )

        topic_str = topic_phrase.format(
            topic=topic, plan_str=plan_str, brainstorm_reminder=brainstorm_reminder
        )
    else:
        topic_str = ""

    stock = args.stock
    if stock:
        stock_str = stock_phrase.format(stock=stock.company_name)
    else:
        stock_str = STOCK_PHRASE_DIVERSITY

    texts_str = GPTTokenizer(DEFAULT_LLM).do_truncation_if_needed(
        texts_str,
        [
            summarize_main_prompt.template,
            summarize_sys_prompt.template,
            chat_str,
            topic_str,
            stock_str,
        ],
    )

    main_prompt = summarize_main_prompt.format(
        texts=texts_str,
        chat_context=chat_str,
        topic_phrase=topic_str,
        stock_phrase=stock_str,
        final_reminder=final_reminder,
        today=(
            context.as_of_date.date().isoformat()
            if context.as_of_date
            else datetime.date.today().isoformat()
        ),
    )

    sys_prompt = summarize_sys_prompt.format(brainstorm_instructions=brainstorm_str)

    result = await llm.do_chat_w_sys_prompt(
        main_prompt,
        sys_prompt,
    )

    if BRAINSTORM_DELIMITER in result:
        result = result.split(BRAINSTORM_DELIMITER)[-1]

    if result.strip() == NO_SUMMARY:  # GPT isn't supposed to do this but does
        return NO_SUMMARY, []

    text, citations = await extract_citations_from_gpt_output(result, text_group, context)
    has_gpt_generated_sources = (
        len([text for text in texts if text.text_type == DEFAULT_TEXT_TYPE]) > 0
    )
    # if we have gpt generated source texts (i.e. this is a "second order" text), then it's reasonable
    # not get to get citations in this round, otherwise we would always expect some
    tries = 0
    while (
        text.strip() != NO_SUMMARY
        and (citations is None or (not has_gpt_generated_sources and len(citations) == 0))
    ) and tries < MAX_CITATION_TRIES:  # failed to load citations, retry
        logger.warning(f"Retrying after no citations after {result}")
        result = await llm.do_chat_w_sys_prompt(
            main_prompt, sys_prompt, no_cache=True, temperature=0.1 * (tries + 1)
        )
        if BRAINSTORM_DELIMITER in result:
            result = result.split(BRAINSTORM_DELIMITER)[-1]

        if result.strip() == NO_SUMMARY:  # GPT isn't supposed to do this but does
            return NO_SUMMARY, []
        text, citations = await extract_citations_from_gpt_output(result, text_group, context)
        tries += 1

    if not text:
        text = result

    if citations is None:
        citations = []
        logger.warning("Text generation failed to produce any citations")

    try:
        citations += await get_second_order_citations(text, get_all_text_citations(texts), context)
    except Exception as e:
        logger.exception(f"Failed to add second order citations: {e}")

    return text, citations


async def _update_summarize_helper(
    args: SummarizeTextInput,
    context: PlanRunContext,
    llm: GPT,
    new_texts: List[Text],
    old_texts: List[Text],
    old_summary: str,
    original_citations: List[TextCitation],
    remaining_original_citation_count: int,
    single_summary: bool = True,
    topic_filter: bool = True,
) -> Tuple[str, List[TextCitation]]:
    logger = get_prefect_logger(__name__)
    last_original_citation_count = get_original_cite_count(original_citations)
    if args.topic and topic_filter:
        new_texts = await topic_filter_helper(
            new_texts,
            args.topic,
            context.agent_id,
            model_for_filter_to_context=GPT4_O,
        )
        if len(new_texts) == 0:
            if remaining_original_citation_count == 0:
                # nothing new or old to cite, return empty summary
                if args.stock:
                    return NO_SUMMARY_FOR_STOCK, []
                else:
                    return NO_SUMMARY, []
            elif last_original_citation_count == remaining_original_citation_count:
                # no change, return old stuff
                return old_summary, original_citations
    new_text_group = TextGroup(val=new_texts)
    new_texts_str: str = await Text.get_all_strs(
        new_text_group, include_header=True, text_group_numbering=True, include_symbols=True
    )  # type: ignore
    old_texts_group = TextGroup(val=old_texts, offset=len(new_texts))
    old_texts_str: str = await Text.get_all_strs(
        old_texts_group, include_header=True, text_group_numbering=True, include_symbols=True
    )  # type: ignore

    if context.chat:
        chat_str = context.chat.get_gpt_input()
    else:
        chat_str = ""
    topic = args.topic
    if topic:
        plan_str = (
            get_psql(skip_commit=context.skip_db_commit)
            .get_execution_plan_for_run(context.plan_run_id)
            .get_formatted_plan(numbered=True)
        )

        topic_str = TOPIC_PHRASE_STR_DEFAULT.format(
            topic=topic, plan_str=plan_str, brainstorm_reminder=""
        )
    else:
        topic_str = ""

    if single_summary:
        brainstorm_instructions = ""
    else:
        brainstorm_instructions = PER_SUMMARIZE_INSTRUCTIONS.format(no_summary=NO_SUMMARY)

    stock = args.stock
    if stock:
        stock_str = STOCK_PHRASE_STR_DEFAULT.format(stock=stock.company_name)
    else:
        stock_str = STOCK_PHRASE_DIVERSITY

    new_texts_str, old_texts_str = GPTTokenizer(DEFAULT_LLM).do_multi_truncation_if_needed(
        [new_texts_str, old_texts_str],
        [
            UPDATE_SUMMARIZE_MAIN_PROMPT.template,
            UPDATE_SUMMARIZE_SYS_PROMPT.template,
            chat_str,
            topic_str,
            stock_str,
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
        stock_phrase=stock_str,
        today=(
            context.as_of_date.date().isoformat()
            if context.as_of_date
            else datetime.date.today().isoformat()
        ),
    )

    sys_prompt = UPDATE_SUMMARIZE_SYS_PROMPT.format(
        plan_delimiter=BRAINSTORM_DELIMITER,
        brainstorm_instructions=brainstorm_instructions,
        remaining_citations=remaining_original_citation_count,
        no_summary=NO_SUMMARY,
    )

    result = await llm.do_chat_w_sys_prompt(
        main_prompt,
        sys_prompt,
    )

    result = result.split(BRAINSTORM_DELIMITER)[-1]  # strip planning section

    if result.strip() == NO_SUMMARY:  # GPT isn't supposed to do this but does
        return NO_SUMMARY, []

    combined_text_group = TextGroup.join(new_text_group, old_texts_group)

    text, citations = await extract_citations_from_gpt_output(result, combined_text_group, context)

    tries = 0
    while (
        text.strip() != NO_SUMMARY
        and (
            citations is None
            or (
                remaining_original_citation_count > 0
                and not (
                    max(
                        remaining_original_citation_count * MAX_CITATION_DROP_RATIO,
                        remaining_original_citation_count - tries,
                    )
                    <= len(citations)
                    <= last_original_citation_count + SUMMARIZE_CITATION_INCREASE_LIMIT
                )
            )
        )
        and tries < MAX_CITATION_TRIES
    ):
        logger.warning(f"Retrying after bad citation count for {result}")
        if citations:
            min_value = max(
                remaining_original_citation_count * 0.5, remaining_original_citation_count - tries
            )
            max_value = last_original_citation_count + SUMMARIZE_CITATION_INCREASE_LIMIT
            if len(citations) < min_value:
                logger.warning(
                    f"Output had {len(citations)} original source citations, "
                    f"but wanted at least {min_value}"
                )
            elif len(citations) > max_value:
                logger.warning(
                    f"Output had {len(citations)} original source citations, "
                    f"but wanted no more than {max_value}"
                )
        else:
            logger.warning("Failed to extract citations")
        result = await llm.do_chat_w_sys_prompt(
            main_prompt,
            sys_prompt,
            no_cache=True,
            temperature=0.1 * (tries + 1),
        )
        result = result.split(BRAINSTORM_DELIMITER)[-1]  # strip planning section

        if result.strip() == NO_SUMMARY:  # GPT isn't supposed to do this but does
            return NO_SUMMARY, []
        text, citations = await extract_citations_from_gpt_output(
            result, combined_text_group, context
        )

        tries += 1

    if text.strip() != NO_SUMMARY and (
        not text
        or citations is None
        or (
            remaining_original_citation_count > 0
            and not (
                max(
                    remaining_original_citation_count * MAX_CITATION_DROP_RATIO,
                    remaining_original_citation_count - tries,
                )
                <= len(citations)
                <= last_original_citation_count + SUMMARIZE_CITATION_INCREASE_LIMIT
            )
        )
    ):
        # if the retries failed and we still ended up with a bad result, just try a new regular summarization
        logger.warning("Failed to do summary update, falling back to from-scratch summary")
        return await _initial_summarize_helper(args, context, llm)

    if citations is None:
        citations = []

    try:
        citations += await get_second_order_citations(
            text, get_all_text_citations(args.texts), context
        )
    except Exception as e:
        logger.exception(f"Failed to add second order citations: {e}")

    return text, citations


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
        await tool_log(log="No text data provided for summarization, skipping", context=context)
        return Text(val=NO_SUMMARY)

    if args.topic:
        topic_filter = await is_topical(args.topic, context)
    else:
        topic_filter = False

    original_texts = set(args.texts)

    args.texts = await partition_to_smaller_text_sizes(args.texts, context)

    text = None
    citations = None

    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "summarize_texts")
        if prev_run_info is not None:
            prev_args = SummarizeTextInput.model_validate_json(prev_run_info.inputs_str)
            prev_texts: List[Text] = await partition_to_smaller_text_sizes(prev_args.texts, context)
            prev_output: Text = prev_run_info.output  # type:ignore
            curr_input_texts = set(args.texts)
            new_texts = list(curr_input_texts - set(prev_texts))
            all_old_citations = cast(List[TextCitation], prev_output.history[0].citations)
            for citation in all_old_citations:
                citation.source_text.reset_id()  # need to do this so we can match them
            remaining_citations = [
                citation
                for citation in all_old_citations
                if citation.source_text in curr_input_texts
                or citation.source_text in original_texts
            ]
            if new_texts or (
                remaining_citations and len(all_old_citations) > len(remaining_citations)
            ):
                old_texts = list(set([citation.source_text for citation in remaining_citations]))
                text, citations = await _update_summarize_helper(
                    args,
                    context,
                    llm,
                    new_texts,
                    old_texts,
                    prev_output.val,
                    all_old_citations,
                    get_original_cite_count(remaining_citations),
                    topic_filter=topic_filter,
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
        text, citations = await _initial_summarize_helper(
            args, context, llm, topic_filter=topic_filter
        )

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

    if len(args.stocks) == 0:
        raise EmptyInputError("Cannot do per stock summarize when no stocks provided")

    if len(args.texts) == 0:
        raise EmptyInputError("Cannot summarize when no texts provided")

    topic_filter = await is_topical(args.topic, context)
    if topic_filter:
        await tool_log("Applying topic filtering before summarization", context)

    original_texts = set(args.texts)

    args.texts = cast(
        List[StockText], await partition_to_smaller_text_sizes(args.texts, context)  # type:ignore
    )

    text_dict: Dict[StockID, List[StockText]] = {stock: [] for stock in args.stocks}
    for text in args.texts:
        try:
            if text.stock_id in text_dict:
                text_dict[text.stock_id].append(text)
        except AttributeError:
            logger.warning("Non-StockText passed to per stock summarize")

    prev_run_dict: Dict[
        StockID, Tuple[List[StockText], List[TextCitation], List[TextCitation], str]
    ] = defaultdict()

    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "per_stock_summarize_texts")
        if prev_run_info is not None:
            prev_args = PerStockSummarizeTextInput.model_validate_json(prev_run_info.inputs_str)
            prev_texts: List[StockText] = prev_args.texts
            prev_texts = cast(
                List[StockText],
                await partition_to_smaller_text_sizes(prev_texts, context),  # type:ignore
            )
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
                for citation in all_old_citations:
                    citation.source_text.reset_id()  # need to do this so we can match them
                remaining_citations = [
                    citation
                    for citation in all_old_citations
                    if citation.source_text in curr_input_texts
                    or citation.source_text in original_texts
                ]
                prev_run_dict[stock] = (
                    new_texts,
                    remaining_citations,
                    all_old_citations,
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
                new_texts, remaining_citations, all_old_citations, old_summary = prev_run_dict[
                    stock
                ]
                old_texts = list(set([citation.source_text for citation in remaining_citations]))
                if new_texts or (
                    remaining_citations and len(remaining_citations) < len(all_old_citations)
                ):
                    tasks.append(
                        _update_summarize_helper(
                            SummarizeTextInput(texts=new_texts, topic=args.topic, stock=stock),  # type: ignore
                            context,
                            llm,
                            new_texts,  # type: ignore
                            old_texts,
                            old_summary,
                            all_old_citations,
                            get_original_cite_count(remaining_citations),
                            single_summary=False,
                            topic_filter=topic_filter,
                        )
                    )
                else:
                    if remaining_citations:  # no new texts, just use old summary
                        tasks.append(identity((old_summary, remaining_citations)))
                    else:  # unless there is nothing to cite, at which point use default
                        tasks.append(identity((NO_SUMMARY_FOR_STOCK, [])))

            else:
                if text_dict[stock]:
                    tasks.append(
                        _initial_summarize_helper(
                            SummarizeTextInput(texts=text_dict[stock], topic=args.topic, stock=stock),  # type: ignore
                            context,
                            llm,
                            single_summary=False,
                            topic_filter=topic_filter,
                        )
                    )
                else:
                    tasks.append(identity((NO_SUMMARY_FOR_STOCK, [])))
            summary_count += 1
        else:
            tasks.append(identity(None))

    await tool_log(
        f"Writing texts for {summary_count} stocks, topic: {args.topic}", context=context
    )

    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)
    output = []

    for stock, (summary, citations) in zip(args.stocks, results):
        if summary == NO_SUMMARY:
            summary = NO_SUMMARY_FOR_STOCK
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


class PerIdeaSummarizeTextInput(ToolArgs):
    ideas: List[Idea]
    texts: List[Text]
    topic_template: str
    column_header: str


@tool(
    description=PER_IDEA_SUMMARIZE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
    reads_chat=True,
    enabled=True,
    enabled_checker_func=ideas_enabled,
)
async def per_idea_summarize_texts(
    args: PerIdeaSummarizeTextInput, context: PlanRunContext
) -> List[Idea]:
    logger = get_prefect_logger(__name__)
    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_LLM)

    if len(args.ideas) == 0:
        raise EmptyInputError("Cannot do per idea summarize when no ideas provided")

    if len(args.texts) == 0:
        raise EmptyInputError("Cannot summarize when no texts provided")

    if IDEA not in args.topic_template:
        raise BadInputError("Input topic template missing IDEA placeholder")

    args.texts = await partition_to_smaller_text_sizes(args.texts, context)

    new_ideas = args.ideas  # default to redoing everything

    old_ideas = []

    prev_run_dict: Dict[Idea, Tuple[List[TextCitation], List[TextCitation], str]] = defaultdict()

    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "per_idea_summarize_texts")
        if prev_run_info is not None:
            prev_args = PerIdeaSummarizeTextInput.model_validate_json(prev_run_info.inputs_str)
            prev_input_ideas: List[Idea] = prev_args.ideas
            old_ideas_lookup = {idea.title: idea for idea in prev_input_ideas}
            prev_input_texts: List[Text] = prev_args.texts
            prev_input_texts = await partition_to_smaller_text_sizes(prev_input_texts, context)
            prev_output_ideas: List[Idea] = prev_run_info.output  # type:ignore
            # get the title to text mapping for the old output texts, the titles are the idea titles
            old_output_history_dict: Dict[str, HistoryEntry] = {
                idea.title: idea.history[-1] for idea in prev_output_ideas
            }
            curr_input_texts = set(args.texts)
            old_input_texts = set(prev_input_texts)
            temp_new_ideas = []
            for idea in args.ideas:
                if idea.title not in old_ideas_lookup:
                    temp_new_ideas.append(idea)
                    continue
                old_summary = cast(str, old_output_history_dict[idea.title].explanation)
                all_old_citations = cast(
                    List[TextCitation], old_output_history_dict[idea.title].citations
                )
                for citation in all_old_citations:
                    citation.source_text.reset_id()  # need to do this so we can match them
                remaining_citations = [
                    citation
                    for citation in all_old_citations
                    if citation.source_text in curr_input_texts
                ]
                prev_run_dict[idea] = (
                    remaining_citations,
                    all_old_citations,
                    old_summary,
                )
            new_texts = list(curr_input_texts - old_input_texts)
            new_ideas = temp_new_ideas

    except Exception as e:
        logger.warning(
            f"Failed attempt to update from previous iteration due to {e}, from scratch fallback"
        )

    tasks = []
    for idea in new_ideas:
        topic = args.topic_template.replace(IDEA, f"'{idea.title}'")
        tasks.append(
            _initial_summarize_helper(
                SummarizeTextInput(texts=args.texts, topic=topic),
                context,
                llm,
                single_summary=False,
            )
        )

    if prev_run_dict:
        for idea, (remaining_citations, all_old_citations, old_summary) in prev_run_dict.items():
            old_texts = list(set([citation.source_text for citation in remaining_citations]))
            if new_texts or (
                remaining_citations and len(remaining_citations) < len(all_old_citations)
            ):
                topic = args.topic_template.replace(IDEA, idea.title)
                tasks.append(
                    _update_summarize_helper(
                        SummarizeTextInput(texts=new_texts, topic=topic),  # type: ignore
                        context,
                        llm,
                        new_texts,  # type: ignore
                        old_texts,
                        old_summary,
                        all_old_citations,
                        get_original_cite_count(remaining_citations),
                        single_summary=False,
                    )
                )
            else:
                if remaining_citations:  # no new texts, just use old summary
                    tasks.append(identity((old_summary, remaining_citations)))
                else:  # unless there is nothing to cite, at which point use default
                    tasks.append(identity((NO_SUMMARY, [])))
            old_ideas.append(idea)

    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)

    final_ideas = []

    for idea, (summary, citations) in zip(new_ideas + old_ideas, results):
        final_ideas.append(
            idea.inject_history_entry(
                HistoryEntry(explanation=summary, title=args.column_header, citations=citations)
            )
        )

    return final_ideas


class PerStockGroupSummarizeTextInput(ToolArgs):
    stock_groups: StockGroups
    texts: List[StockText]
    topic_template: str
    column_header: str = "Summary"


@tool(
    description=PER_STOCK_GROUP_SUMMARIZE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
    reads_chat=True,
)
async def per_stock_group_summarize_texts(
    args: PerStockGroupSummarizeTextInput, context: PlanRunContext
) -> StockGroups:
    logger = get_prefect_logger(__name__)
    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_LLM)

    if len(args.texts) == 0:
        raise EmptyInputError("Cannot summarize when no texts provided")

    if f"{STOCK_TYPE} stocks" not in args.topic_template:
        raise BadInputError("Input topic template missing STOCK_TYPE placeholder")

    args.texts = cast(
        List[StockText],
        await partition_to_smaller_text_sizes(cast(List[Text], args.texts), context),
    )

    new_groups = args.stock_groups.stock_groups  # default to redoing everything

    old_groups = []

    prev_run_dict: Dict[StockGroup, Tuple[List[TextCitation], List[TextCitation], str]] = (
        defaultdict()
    )

    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "per_stock_group_summarize_texts")
        if prev_run_info is not None:
            prev_args = PerStockGroupSummarizeTextInput.model_validate_json(
                prev_run_info.inputs_str
            )
            prev_input_texts: List[StockText] = prev_args.texts
            prev_input_texts = cast(
                List[StockText],
                await partition_to_smaller_text_sizes(cast(List[Text], prev_input_texts), context),
            )
            prev_output_groups: StockGroups = prev_run_info.output.stock_group  # type:ignore
            old_output_groups_lookup: Dict[str, StockGroup] = {
                group.name: group for group in prev_output_groups.stock_groups
            }
            curr_input_texts = set(args.texts)
            old_input_texts = set(prev_input_texts)
            new_texts = list(curr_input_texts - old_input_texts)
            prelim_new_groups = []
            for stock_group in args.stock_groups.stock_groups:
                if stock_group.name not in old_output_groups_lookup:
                    prelim_new_groups.append(stock_group)
                    continue
                old_output_group = old_output_groups_lookup[stock_group.name]
                old_summary: str = cast(str, old_output_group.history[-1].explanation)
                all_old_citations = cast(List[TextCitation], old_output_group.history[-1].citations)
                for citation in all_old_citations:
                    citation.source_text.reset_id()  # need to do this so we can match them
                remaining_citations = [
                    citation
                    for citation in all_old_citations
                    if citation.source_text in curr_input_texts
                ]
                prev_run_dict[stock_group] = (
                    remaining_citations,
                    all_old_citations,
                    old_summary,
                )
            new_groups = prelim_new_groups

    except Exception as e:
        logger.warning(
            f"Failed attempt to update from previous iteration due to {e}, from scratch fallback"
        )

    tasks = []
    for group in new_groups:
        group_set = set(group.stocks)
        if group.ref_stock:
            group_set.add(group.ref_stock)  # include text data from target stock if any
        topic = (
            (args.topic_template.replace(STOCK_TYPE, group.name))
            if "Competitors" not in group.name
            else args.topic_template.replace("STOCK_TYPE stocks", group.name)
        )
        if group.ref_stock:
            topic = topic.replace(
                TARGET_STOCK,
                (
                    group.ref_stock.company_name
                    if group.ref_stock.company_name
                    else group.ref_stock.isin
                ),
            )
        group_texts = [text for text in args.texts if text.stock_id and text.stock_id in group_set]
        tasks.append(
            _initial_summarize_helper(
                SummarizeTextInput(texts=cast(List[Text], group_texts), topic=topic),
                context,
                llm,
                single_summary=False,
            )
        )

    if prev_run_dict:
        for group, (remaining_citations, all_old_citations, old_summary) in prev_run_dict.items():
            group_set = set(group.stocks)
            if group.ref_stock:
                group_set.add(group.ref_stock)  # include text data from target stock if any
            group_new_texts = [
                text for text in new_texts if text.stock_id and text.stock_id in group_set
            ]
            old_texts = list(set([citation.source_text for citation in remaining_citations]))
            if group_new_texts or (
                remaining_citations and len(remaining_citations) < len(all_old_citations)
            ):
                topic = (
                    (args.topic_template.replace(STOCK_TYPE, group.name))
                    if "Competitors" not in group.name
                    else args.topic_template.replace("STOCK_TYPE stocks", group.name)
                )
                if group.ref_stock:
                    topic = topic.replace(
                        TARGET_STOCK,
                        (
                            group.ref_stock.company_name
                            if group.ref_stock.company_name
                            else group.ref_stock.isin
                        ),
                    )
                tasks.append(
                    _update_summarize_helper(
                        SummarizeTextInput(texts=group_new_texts, topic=topic),  # type: ignore
                        context,
                        llm,
                        new_texts,  # type: ignore
                        old_texts,
                        old_summary,
                        all_old_citations,
                        get_original_cite_count(remaining_citations),
                        single_summary=False,
                    )
                )
            else:
                if remaining_citations:  # no new texts, just use old summary
                    tasks.append(identity((old_summary, remaining_citations)))
                else:  # unless there is nothing to cite, at which point use default
                    tasks.append(identity((NO_SUMMARY, [])))
            old_groups.append(group)

    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)
    output = []

    for group, (summary, citations) in zip(new_groups + old_groups, results):
        group = group.inject_history_entry(
            HistoryEntry(explanation=summary, title=args.column_header, citations=citations)
        )
        output.append(group)

    return StockGroups(header=args.stock_groups.header, stock_groups=output)


class CompareTextInput(ToolArgs):
    group1: List[Text]
    group1_label: str
    group2: List[Text]
    group2_label: str
    extra_data: Optional[List[Text]] = None
    extra_data_label: Optional[str] = None

    # prompt arguments (hidden from planner)
    comparison_sys_prompt: str = COMPARISON_SYS_PROMPT_STR_DEFAULT
    comparison_main_prompt: str = COMPARISON_MAIN_PROMPT_STR_DEFAULT

    # tool arguments metadata
    arg_metadata = {
        "comparison_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
        "comparison_main_prompt": ToolArgMetadata(hidden_from_planner=True),
    }


@tool(
    description=COMPARISON_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
    reads_chat=True,
    update_instructions=COMPARISON_UPDATE_INSTRUCTIONS,
)
async def compare_texts(args: CompareTextInput, context: PlanRunContext) -> Text:
    logger = get_prefect_logger(__name__)
    COMPARISON_SYS_PROMPT_TEMPLATE = args.comparison_sys_prompt + CITATION_PROMPT

    COMPARISON_MAIN_PROMPT_TEMPLATE = (
        args.comparison_main_prompt
        + CITATION_REMINDER
        + "\nNow write your comparison, with citations:\n"
    )

    COMPARISON_SYS_PROMPT = Prompt(
        name="LLM_COMPARISON_SYS_PROMPT",
        template=COMPARISON_SYS_PROMPT_TEMPLATE,
    )

    COMPARISON_MAIN_PROMPT = Prompt(
        name="LLM_COMPARISON_MAIN_PROMPT",
        template=COMPARISON_MAIN_PROMPT_TEMPLATE,
    )

    if context.chat is None:
        # just for mypy, and shouldn't happen
        return Text(val="")
    # TODO we need guardrails on this

    if len(args.group1) == 0:
        raise EmptyInputError("Cannot compare when no text provided for group 1")
    if len(args.group2) == 0:
        raise EmptyInputError("Cannot compare when no text provided for group 2")

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_LLM)
    text_group1 = TextGroup(val=args.group1)
    group1_str: str = await Text.get_all_strs(
        text_group1, include_header=True, include_symbols=True, text_group_numbering=True
    )  # type: ignore
    text_group2 = TextGroup(val=args.group2, offset=len(args.group1))
    group2_str: str = await Text.get_all_strs(
        text_group2, include_header=True, include_symbols=True, text_group_numbering=True
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
    main_prompt = COMPARISON_MAIN_PROMPT.format(
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
    )

    sys_prompt = COMPARISON_SYS_PROMPT.format()

    result = await llm.do_chat_w_sys_prompt(main_prompt=main_prompt, sys_prompt=sys_prompt)

    merged_group = TextGroup.join(text_group1, text_group2)
    text, citations = await extract_citations_from_gpt_output(result, merged_group, context)

    tries = 0
    while citations is None and tries < MAX_CITATION_TRIES:  # failed to load citations, retry
        logger.warning(f"Retrying after no citations after  {result}")
        result = await llm.do_chat_w_sys_prompt(
            main_prompt, sys_prompt, no_cache=True, temperature=0.1 * (tries + 1)
        )
        text, citations = await extract_citations_from_gpt_output(result, merged_group, context)
        tries += 1

    if not text:
        text = result

    if citations is None:
        citations = []
        logger.warning("Text generation failed to produce any citations")

    try:
        citations += await get_second_order_citations(
            text, get_all_text_citations(args.group1 + args.group2), context
        )
    except Exception as e:
        logger.exception(f"Failed to add second order citations: {e}")

    comparison: Text = Text(val=text)
    comparison = comparison.inject_history_entry(
        HistoryEntry(title="Text Comparison", citations=citations)  # type: ignore
    )
    return comparison


class AnswerQuestionInput(ToolArgs):
    question: str
    texts: List[Text]

    # prompt arguments (hidden from planner)
    answer_question_main_prompt: str = ANSWER_QUESTION_MAIN_PROMPT_STR_DEFAULT
    answer_question_sys_prompt: str = ANSWER_QUESTION_SYS_PROMPT_STR_DEFAULT

    # tool arguments metadata
    arg_metadata = {
        "answer_question_main_prompt": ToolArgMetadata(hidden_from_planner=True),
        "answer_question_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
    }


@tool(
    description=ANSWER_QUESTION_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
)
async def answer_question_with_text_data(
    args: AnswerQuestionInput, context: PlanRunContext
) -> Text:
    logger = get_prefect_logger(__name__)
    answer_question_main_prompt = Prompt(
        name="LLM_ANSWER_QUESTION_MAIN_PROMPT",
        template=args.answer_question_main_prompt
        + CITATION_REMINDER
        + " Now write your answer, with citations:\n",
    )
    answer_question_sys_prompt = Prompt(
        name="LLM_ANSWER_QUESTION_SYS_PROMPT",
        template=args.answer_question_sys_prompt + CITATION_PROMPT,
    )

    if len(args.texts) == 0:
        raise EmptyInputError(message="Cannot answer question with an empty list of texts")
    # TODO we need guardrails on this
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_LLM)
    text_group = TextGroup(val=args.texts)
    texts_str: str = await Text.get_all_strs(  # type: ignore
        text_group, include_header=True, include_symbols=True, text_group_numbering=True
    )
    texts_str = GPTTokenizer(DEFAULT_LLM).do_truncation_if_needed(
        texts_str,
        [answer_question_main_prompt.template, answer_question_sys_prompt.template, args.question],
    )

    main_prompt = answer_question_main_prompt.format(
        texts=texts_str,
        question=args.question,
        today=(
            context.as_of_date.date().isoformat()
            if context.as_of_date
            else datetime.date.today().isoformat()
        ),
    )

    sys_prompt = answer_question_sys_prompt.format()

    result = await llm.do_chat_w_sys_prompt(main_prompt=main_prompt, sys_prompt=sys_prompt)
    text, citations = await extract_citations_from_gpt_output(result, text_group, context)

    tries = 0
    while citations is None and tries < MAX_CITATION_TRIES:  # failed to load citations, retry
        logger.warning(f"Retrying after no citations after  {result}")
        result = await llm.do_chat_w_sys_prompt(
            main_prompt, sys_prompt, no_cache=True, temperature=0.1 * (tries + 1)
        )
        text, citations = await extract_citations_from_gpt_output(result, text_group, context)
        tries += 1

    if not text:
        text = result

    if citations is None:
        citations = []
        logger.warning("Text generation failed to produce any citations")

    try:
        citations += await get_second_order_citations(
            text, get_all_text_citations(args.texts), context
        )
    except Exception as e:
        logger.exception(f"Failed to add second order citations: {e}")

    answer: Text = Text(val=text)
    answer = answer.inject_history_entry(
        HistoryEntry(title="Summary", citations=citations)  # type:ignore
    )
    return answer


# Topic filter


async def topic_filter_helper(
    texts: List[Text],
    topic: str,
    agent_id: str,
    model_for_filter_to_context: Optional[str] = None,
    no_empty: bool = False,
) -> List[Text]:

    # sort first by timestamp so more likely to drop older, less relevant texts
    texts.sort(key=lambda x: x.timestamp.timestamp() if x.timestamp else 0, reverse=True)

    text_strs: List[str] = await Text.get_all_strs(texts, include_header=True, include_symbols=True)  # type: ignore
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=GPT4_O_MINI)
    tokenizer = GPTTokenizer(GPT4_O_MINI)
    used = tokenizer.get_token_length(
        "\n".join([TOPIC_FILTER_MAIN_PROMPT.template, TOPIC_FILTER_SYS_PROMPT.template, topic])
    )
    tasks = []
    for text_str in text_strs:
        text_str = tokenizer.chop_input_to_allowed_length(text_str, used)
        tasks.append(
            llm.do_chat_w_sys_prompt(
                TOPIC_FILTER_MAIN_PROMPT.format(text=text_str, topic=topic),
                TOPIC_FILTER_SYS_PROMPT.format(),
            )
        )
    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)

    token_counts: Dict[int, int] = defaultdict(int)
    all_tokens_count = 0
    score_tuples = []
    for result, text_str in zip(results, text_strs):

        try:
            rationale, score = result.strip().replace("\n\n", "\n").split("\n")
            score = int(score)
        except ValueError:
            score = 0
            rationale = "No relevance"

        # this works under assumption that both LLM types have same tokenizer, true for GPT4O models
        text_tokens = tokenizer.get_token_length(text_str)
        token_counts[score] += text_tokens
        all_tokens_count += text_tokens
        score_tuples.append((score, rationale))

    token_filter_upper_bound = LLM_FILTER_MAX_PERCENT * all_tokens_count
    if model_for_filter_to_context:
        token_filter_upper_bound = max(
            token_filter_upper_bound,
            MAX_TOKENS[model_for_filter_to_context] * LLM_FILTER_MAX_INPUT_PERCENTAGE,
        )
    token_filter_lower_bound = LLM_FILTER_MIN_TOKENS

    if token_counts[3] > token_filter_upper_bound:
        # If there are lot of 3s, only include 3
        # these may need to be truncated, but not here, still support basic topic filtering case
        answers = [(score >= 3, rationale) for score, rationale in score_tuples]
    else:
        if token_counts[3] + token_counts[2] < token_filter_lower_bound:
            # If there are hardly any 3 + 2, include 1s, optionally
            cutoff = 1
        else:
            # otherwise, includes 3s and 2s, optionally
            cutoff = 2

        # we always include all texts above the cutoff, include texts at the cutoff as long as
        # we have tokens in our quota
        must_include_token_count = sum([token_counts[i] for i in range(cutoff + 1, 4)])
        optional_token_counts = 0
        answers = []
        for (score, rationale), text_str in zip(score_tuples, text_strs):
            if score < cutoff:
                answers.append((False, rationale))
            elif score > cutoff:
                answers.append((True, rationale))
            else:
                optional_token_counts += tokenizer.get_token_length(text_str)
                if optional_token_counts + must_include_token_count < token_filter_upper_bound:
                    answers.append((True, rationale))
                else:
                    answers.append((False, rationale))

    output = [
        text.inject_history_entry(HistoryEntry(explanation=reason, title=f"Connection to {topic}"))
        for text, (is_relevant, reason) in zip(texts, answers)
        if is_relevant
    ]
    if (
        not output and no_empty
    ):  # we filtered everything, but that's not allowed, so just return input
        return texts

    return output


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

    return await topic_filter_helper(args.news_texts, args.topic, context.agent_id)  # type: ignore


# Profile filter


async def profile_filter_helper(
    aligned_text_groups: StockAlignedTextGroups,
    str_lookup: Dict[StockID, str],
    profile: str,
    is_using_complex_profile: bool,
    llm: GPT,
    profile_filter_main_prompt: Prompt,
    context: PlanRunContext,
    simple_profile_filter_sys_prompt: Optional[Prompt] = None,
    complex_profile_filter_sys_prompt: Optional[Prompt] = None,
    topic: str = "",
    do_citations: bool = True,
    stock_whitelist: Optional[Set[StockID]] = None,
) -> List[Tuple[bool, str, List[Citation]]]:
    logger = get_prefect_logger(__name__)
    if (simple_profile_filter_sys_prompt is None) and (complex_profile_filter_sys_prompt is None):
        logger.error(
            "Simple Filter System Prompt and Complex Filter System Prompt cannot both be None!"
        )

    # Add a placeholder company name as a safety buffer
    tokenizer = GPTTokenizer(GPT4_O)
    used = 0
    if is_using_complex_profile and isinstance(complex_profile_filter_sys_prompt, Prompt):
        used = tokenizer.get_token_length(
            "\n".join(
                [
                    profile_filter_main_prompt.template,
                    complex_profile_filter_sys_prompt.template,
                    topic,
                    profile,
                    "Placeholder Company Name",
                ]
            )
        )
    elif isinstance(simple_profile_filter_sys_prompt, Prompt):
        used = tokenizer.get_token_length(
            "\n".join(
                [
                    profile_filter_main_prompt.template,
                    simple_profile_filter_sys_prompt.template,
                    profile,
                    "Placeholder Company Name",
                ]
            )
        )

    tasks = []
    for stock in aligned_text_groups.val:
        text_str = str_lookup[stock]
        text_str = tokenizer.chop_input_to_allowed_length(text_str, used)
        if text_str == "" or (stock_whitelist is not None and stock not in stock_whitelist):
            tasks.append(identity(""))

        elif is_using_complex_profile and isinstance(complex_profile_filter_sys_prompt, Prompt):
            tasks.append(
                llm.do_chat_w_sys_prompt(
                    profile_filter_main_prompt.format(
                        company_name=stock.company_name,
                        texts=text_str,
                        profile=profile,
                        today=(
                            context.as_of_date.date().isoformat()
                            if context.as_of_date
                            else datetime.date.today().isoformat()
                        ),
                    ),
                    complex_profile_filter_sys_prompt.format(topic_name=topic),
                )
            )
        elif isinstance(simple_profile_filter_sys_prompt, Prompt):
            tasks.append(
                llm.do_chat_w_sys_prompt(
                    profile_filter_main_prompt.format(
                        company_name=stock.company_name,
                        texts=text_str,
                        profile=profile,
                        today=(
                            context.as_of_date.date().isoformat()
                            if context.as_of_date
                            else datetime.date.today().isoformat()
                        ),
                    ),
                    simple_profile_filter_sys_prompt.format(),
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
        {stock: stock_text_diff.get(stock, []) for stock in added_stocks},
        include_header=True,
        text_group_numbering=False,
    )

    tasks = []
    for stock in added_stocks:
        if str_lookup[stock]:
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
        else:
            tasks.append(identity(""))

    results = await gather_with_concurrency(tasks)
    return {
        stock: explanation.split("\n")[0]
        for stock, explanation in zip(added_stocks, results)
        if "Yes, " in explanation
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
        {stock: stock_text_diff.get(stock, []) for stock in removed_stocks},
        include_header=True,
        text_group_numbering=False,
    )

    tasks = []
    for stock in removed_stocks:
        if str_lookup[stock]:
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
        else:
            tasks.append(identity(""))

    results = await gather_with_concurrency(tasks)
    return {
        stock: explanation.split("\n")[0]
        for stock, explanation in zip(removed_stocks, results)
        if "Agreed, " in explanation
    }


async def run_profile_match(
    stocks: List[StockID],
    profile: Union[str, TopicProfiles],
    texts: List[StockText],
    profile_filter_main_prompt_str: str,
    profile_filter_sys_prompt_str: str,
    profile_output_instruction_str: str,
    context: PlanRunContext,
    use_cache: bool = True,
    detailed_log: bool = True,
    crash_on_empty: bool = True,
    debug_info: Optional[Dict[str, Any]] = None,
) -> List[StockID]:

    logger = get_prefect_logger(__name__)
    if context.task_id is None:
        return []  # for mypy

    profile_str: str = ""
    if isinstance(profile, TopicProfiles):
        is_using_complex_profile = True
        profile_str = await Text.get_all_strs(  # type: ignore
            profile, include_header=False, text_group_numbering=False
        )
        await tool_log(
            f"Filtering stocks for advanced profile with topic: {profile.topic}", context=context
        )
    elif isinstance(profile, str):
        is_using_complex_profile = False
        profile_str = profile
        await tool_log(f"Filtering stocks for simple profile: {profile_str}", context=context)
    else:
        raise ValueError(
            "profile must be either a string or a TopicProfiles object "
            "in filter_stocks_by_profile_match function!"
        )

    # Save the original text ids to compare against the prev texts, no need to
    # save the whole text object
    original_text = set(texts)
    texts = await partition_to_smaller_text_sizes(texts, context=context)  # type: ignore
    split_texts_set = set(texts)
    filtered_down_texts = await classify_stock_text_relevancies_for_profile(
        texts, profiles_str=profile_str, context=context  # type: ignore
    )

    # Adding this so we can compare mini filtering with other filtering methods
    if debug_info is not None:
        debug_info["filtered_texts"] = dump_io_type(filtered_down_texts)

    # TODO: This needs to be moved outside the profile match helper function
    prev_run_info = None
    try:  # since everything associated with diffing is optional, put in try/except
        if use_cache:
            prev_run_info = await get_prev_run_info(context, "filter_stocks_by_profile_match")
        else:
            prev_run_info = None
        if prev_run_info is not None:
            prev_args = FilterStocksByProfileMatch.model_validate_json(prev_run_info.inputs_str)
            prev_output_stocks: List[StockID] = prev_run_info.output  # type:ignore

            # start by finding the differences in the input texts for the two runs,
            # need to start by partitioning and reclassifying the previous texts
            partitioned_prev_texts = cast(
                List[StockText],
                await partition_to_smaller_text_sizes(
                    prev_args.texts, context=context  # type: ignore
                ),
            )
            prev_stock_text_lookup = get_stock_text_lookup(partitioned_prev_texts)
            current_stock_text_lookup = get_stock_text_lookup(filtered_down_texts)

            prev_stock_id_by_gbi_lookup = {stock.gbi_id: stock for stock in prev_output_stocks}

            diff_text_stocks = []
            same_text_stocks = []
            stock_text_diff = {}
            current_input_set = set(stocks)
            for stock in stocks:
                added_text_diff = get_text_diff(
                    current_stock_text_lookup.get(stock, []), prev_stock_text_lookup.get(stock, [])
                )
                if added_text_diff:
                    diff_text_stocks.append(stock)
                    if added_text_diff:
                        stock_text_diff[stock] = added_text_diff
                else:  # redo ones that have citations that are missing
                    output_stock = prev_stock_id_by_gbi_lookup.get(stock.gbi_id)
                    if output_stock:
                        missing_citations = False
                        for citation in output_stock.history[-1].citations:
                            if isinstance(citation, TextCitation):
                                citation.source_text.reset_id()
                                if (citation.source_text not in split_texts_set) and (
                                    citation.source_text not in original_text
                                ):
                                    missing_citations = True
                                    break

                        if missing_citations:
                            diff_text_stocks.append(stock)
                        else:
                            same_text_stocks.append(stock)

            # we are only going to do main pass for stocks that have some text difference
            # relative to previous run
            stocks = diff_text_stocks

    except Exception as e:
        logger.warning(f"Error doing text diff from previous run: {e}")

    stocks_with_texts: List[StockID] = []
    gbi_ids_with_texts = set([text.stock_id.gbi_id for text in filtered_down_texts])  # type: ignore
    for stock in stocks:
        if stock.gbi_id in gbi_ids_with_texts:
            stocks_with_texts.append(stock)

    if detailed_log:
        no_info_stock_count = len(stocks) - len(stocks_with_texts)
        if no_info_stock_count > 0:
            await tool_log(
                f"No new relevant information for {no_info_stock_count} stocks, skipping these stocks",
                context=context,
            )

    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(
        stocks_with_texts, filtered_down_texts
    )

    str_lookup: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val, include_header=True, text_group_numbering=True
    )

    # prepare prompts
    profile_filter_main_prompt = Prompt(
        name="PROFILE_FILTER_MAIN_PROMPT",
        template=(
            profile_filter_main_prompt_str
            + CITATION_REMINDER
            + " Now discuss your decision in a single paragraph, "
            + "provide a final answer, and then an anchor mapping json:\n"
        ),
    )

    if is_using_complex_profile:
        profile_filter_sys_prompt = Prompt(
            name="COMPLEX_PROFILE_FILTER_SYS_PROMPT",
            template=profile_filter_sys_prompt_str
            + profile_output_instruction_str
            + CITATION_PROMPT,
        )
    else:
        profile_filter_sys_prompt = Prompt(
            name="SIMPLE_PROFILE_FILTER_SYS_PROMPT",
            template=(
                profile_filter_sys_prompt_str + profile_output_instruction_str + CITATION_PROMPT
            ),
        )

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    cheap_llm = GPT(context=gpt_context, model=GPT4_O_MINI)
    stock_whitelist: Set[StockID] = set()
    if detailed_log:
        await tool_log(
            f"Starting filtering with {len(aligned_text_groups.val.keys())} stocks",
            context=context,
        )

    if is_using_complex_profile:
        simple_profile_filter_sys_prompt = None
        complex_profile_filter_sys_prompt = profile_filter_sys_prompt
    else:
        simple_profile_filter_sys_prompt = profile_filter_sys_prompt
        complex_profile_filter_sys_prompt = None

    for stock, (is_relevant, reason, citations) in zip(
        aligned_text_groups.val.keys(),
        await profile_filter_helper(
            aligned_text_groups,
            str_lookup,
            profile_str,
            is_using_complex_profile,
            llm=cheap_llm,
            profile_filter_main_prompt=profile_filter_main_prompt,
            simple_profile_filter_sys_prompt=simple_profile_filter_sys_prompt,
            complex_profile_filter_sys_prompt=complex_profile_filter_sys_prompt,
            context=context,
            do_citations=False,
        ),
    ):
        if is_relevant:
            stock_whitelist.add(stock)

    if detailed_log:
        await tool_log(
            f"Completed a surface level round of filtering. {len(stock_whitelist)} stocks remaining.",
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
                profile_filter_main_prompt=profile_filter_main_prompt,
                simple_profile_filter_sys_prompt=simple_profile_filter_sys_prompt,
                complex_profile_filter_sys_prompt=complex_profile_filter_sys_prompt,
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

    if detailed_log:
        await tool_log(
            f"Completed a more in-depth round of filtering. {len(filtered_stocks)} stocks remaining.",
            context=context,
        )

    # No need for an else since we can guarantee at this point one is not None, appeases linter
    if isinstance(profile, TopicProfiles):
        # TODO: Update the rubric to handle the new extensive profile data we have, for now
        # we just pass in the topic which is a short simple string similar to a profile string
        profile_data_for_rubric = profile.topic
    elif isinstance(profile, str):
        profile_data_for_rubric = profile

    rubric_dict = await get_profile_rubric(profile_data_for_rubric, context.agent_id)
    # Assigns scores inplace
    filtered_stocks_with_scores = await stocks_rubric_score_assignment(
        filtered_stocks, rubric_dict, stock_reason_map, profile_data_for_rubric, context
    )

    try:  # since everything associated with diffing is optional, put in try/except
        if prev_run_info is not None:
            if context.diff_info is not None:  # collect info for automatic diff
                prev_input_set = set(prev_args.stocks)
                prev_output_set = set(prev_output_stocks)
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
                    stock
                    for stock in filtered_stocks_with_scores
                    if stock not in added_stocks or stock in added_diff_info
                ]

                current_output_set = set(filtered_stocks_with_scores)

                removed_stocks = []

                # identify stocks that didn't make it through this pass of the tool but did last time

                for stock in prev_output_stocks:
                    if (
                        stock in current_input_set
                        and stock not in current_output_set
                        and stock not in same_text_stocks
                    ):
                        removed_stocks.append(stock)

                removed_diff_info = await profile_filter_removed_diff_info(
                    removed_stocks, profile_str, stock_text_diff, context.agent_id
                )

                # don't remove stocks where we couldn't get a good diff explanation
                # as long as we also didn't lose citations

                for old_stock in removed_stocks:
                    if old_stock not in removed_diff_info:
                        for new_stock in current_input_set:
                            if new_stock == old_stock:
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
                            removed_diff_info[old_stock] = NO_CITATIONS_DIFF

                context.diff_info[context.task_id] = {
                    "added": added_diff_info,
                    "removed": removed_diff_info,
                }

    except Exception as e:
        logger.warning(f"Error doing text diff from previous run: {e}")

    try:  # we do this part separately because we don't want to accidently lose the same_text_stocks
        if prev_run_info is not None:
            if same_text_stocks:  # add in results where no change in text
                old_pass_count = 0
                for new_stock in same_text_stocks:
                    found_stock = False
                    for old_stock in prev_output_stocks:
                        if old_stock == new_stock:
                            found_stock = True
                            break
                    if found_stock:
                        stock_with_old_history = add_old_history(
                            new_stock, old_stock, context.task_id
                        )
                        if stock_with_old_history:
                            old_pass_count += 1
                            filtered_stocks_with_scores.append(stock_with_old_history)

                if old_pass_count > 0:
                    if detailed_log:
                        await tool_log(
                            f"Including {old_pass_count} stocks that have no update and passed filter previously",
                            context=context,
                        )

    except Exception as e:
        logger.warning(f"Error duplicating output for stocks with no text changes: {e}")

    if isinstance(profile, TopicProfiles):
        await tool_log(
            f"{len(filtered_stocks_with_scores)} stocks passed filter for profile: {profile.topic}",
            context=context,
        )
    elif isinstance(profile, str):
        await tool_log(
            f"{len(filtered_stocks_with_scores)} stocks passed filter stocks for profile: {profile_str}",
            context=context,
        )

    if not filtered_stocks_with_scores and crash_on_empty:
        raise EmptyOutputError(
            message=f"Stock profile filter looking for '{profile_str}' resulted in an empty list of stocks"
        )
    # dedup stocks
    company_names = set()
    dedup_res = []
    for stock in filtered_stocks_with_scores:
        if stock.company_name not in company_names:
            company_names.add(stock.company_name)
            dedup_res.append(stock)
    dedup_res = sorted(
        dedup_res, key=lambda stock: stock.history[-1].score.val, reverse=True  # type: ignore
    )
    return dedup_res


class FilterStocksByProfileMatch(ToolArgs):
    stocks: List[StockID]
    texts: List[StockText]
    profile: Union[str, TopicProfiles]

    # prompt arguments (hidden from planner)
    profile_filter_main_prompt: str = PROFILE_FILTER_MAIN_PROMPT_STR_DEFAULT
    simple_profile_filter_sys_prompt: str = SIMPLE_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT
    complex_profile_filter_sys_prompt: str = COMPLEX_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT
    profile_output_instruction: str = PROFILE_OUTPUT_INSTRUCTIONS_DEFAULT

    # tool arguments metadata
    arg_metadata = {
        "profile_filter_main_prompt": ToolArgMetadata(hidden_from_planner=True),
        "simple_profile_filter_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
        "complex_profile_filter_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
        "profile_output_instruction": ToolArgMetadata(hidden_from_planner=True),
    }


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

    profile_filter_sys_prompt = ""
    if isinstance(args.profile, str):
        profile_filter_sys_prompt = args.simple_profile_filter_sys_prompt
    elif isinstance(args.profile, TopicProfiles):
        profile_filter_sys_prompt = args.complex_profile_filter_sys_prompt
    else:
        raise ValueError(
            "profile must be either a string or a TopicProfiles object "
            "in filter_stocks_by_profile_match function!"
        )

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    return await run_profile_match(
        stocks=args.stocks,
        profile=args.profile,
        texts=args.texts,
        profile_filter_main_prompt_str=args.profile_filter_main_prompt,
        profile_filter_sys_prompt_str=profile_filter_sys_prompt,
        profile_output_instruction_str=args.profile_output_instruction,
        context=context,
        use_cache=True,
        debug_info=debug_info,
    )


class PerIdeaFilterStocksByProfileMatch(ToolArgs):
    stocks: List[StockID]
    texts: List[StockText]
    ideas: List[Idea]
    profiles: List[TopicProfiles]

    # prompt arguments (hidden from planner)
    profile_filter_main_prompt: str = PROFILE_FILTER_MAIN_PROMPT_STR_DEFAULT
    simple_profile_filter_sys_prompt: str = SIMPLE_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT
    complex_profile_filter_sys_prompt: str = COMPLEX_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT
    profile_output_instruction: str = PROFILE_OUTPUT_INSTRUCTIONS_DEFAULT

    # tool arguments metadata
    arg_metadata = {
        "profile_filter_main_prompt": ToolArgMetadata(hidden_from_planner=True),
        "simple_profile_filter_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
        "complex_profile_filter_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
        "profile_output_instruction": ToolArgMetadata(hidden_from_planner=True),
    }


@tool(
    description=PER_TOPIC_FILTER_BY_PROFILE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
    enabled=True,
    enabled_checker_func=ideas_enabled,
)
async def per_idea_filter_stocks_by_profile_match(
    args: PerIdeaFilterStocksByProfileMatch,
    context: PlanRunContext,
) -> StockGroups:
    logger = get_prefect_logger(__name__)
    if len(args.stocks) == 0:
        raise EmptyInputError("Cannot filter empty list of stocks")
    if len(args.texts) == 0:
        raise EmptyInputError("Cannot filter stocks with empty list of texts")
    if len(args.profiles) == 0:
        raise EmptyInputError("Cannot filter stocks with empty list of profiles")
    if len(args.ideas) == 0:
        raise EmptyInputError("Cannot filter stocks with empty list of ideas")

    profile_filter_sys_prompt = args.complex_profile_filter_sys_prompt

    profile_idea_lookup: Dict[str, Idea] = {}
    for profile in args.profiles:
        for idea in args.ideas:
            if profile.initial_idea == idea.title:
                profile_idea_lookup[profile.initial_idea] = idea

    prev_run_info = None

    todo_profiles = args.profiles[:]

    removed_stocks: Set[StockID] = set()
    cached_profiles: List[TopicProfiles] = []
    cached_filtered_stocks = {}

    # TODO: doing this both here and inside run_profile_match is redundant, but basically has no
    # effect, and useful to have them available for checking against citations
    split_texts = cast(
        List[StockText], await partition_to_smaller_text_sizes(args.texts, context=context)  # type: ignore
    )

    tasks = []

    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "per_idea_filter_stocks_by_profile_match")
        if prev_run_info is not None:
            prev_args = PerIdeaFilterStocksByProfileMatch.model_validate_json(
                prev_run_info.inputs_str
            )
            old_ideas = set(prev_args.ideas)
            old_stocks = set(prev_args.stocks)
            new_stocks = set(args.stocks)
            added_stocks = new_stocks - old_stocks
            removed_stocks = old_stocks - new_stocks
            all_texts = set(args.texts) | set(split_texts)
            prev_output: StockGroups = prev_run_info.output  # type:ignore
            prev_group_lookup = {group.name: group for group in prev_output.stock_groups}
            using_cache_count = 0
            for profile in args.profiles:
                if (
                    profile.initial_idea in profile_idea_lookup
                    and profile_idea_lookup[profile.initial_idea] in old_ideas
                    and profile.topic in prev_group_lookup
                ):
                    must_do_stocks = list(added_stocks)
                    for stock in prev_group_lookup[profile.topic].stocks:
                        for citation in stock.history[-1].citations:
                            if isinstance(citation, TextCitation):
                                citation.source_text.reset_id()
                                if citation.source_text not in all_texts:
                                    must_do_stocks.append(stock)
                                    break

                    if must_do_stocks:
                        tasks.append(
                            run_profile_match(
                                stocks=must_do_stocks,
                                profile=profile,
                                texts=split_texts,
                                profile_filter_main_prompt_str=args.profile_filter_main_prompt,
                                profile_filter_sys_prompt_str=profile_filter_sys_prompt,
                                profile_output_instruction_str=args.profile_output_instruction,
                                context=context,
                                use_cache=False,
                                crash_on_empty=False,
                            )
                        )
                    else:
                        tasks.append(identity([]))

                    old_group = prev_group_lookup[profile.topic]
                    cached_filtered_stocks[profile.initial_idea] = [
                        stock
                        for stock in old_group.stocks
                        if stock not in removed_stocks and stock not in must_do_stocks
                    ]
                    if len(cached_filtered_stocks[profile.initial_idea]) > 0:
                        using_cache_count += 1
                    todo_profiles.remove(profile)
                    cached_profiles.append(profile)

            if using_cache_count > 0:
                await tool_log(
                    f"Using previous run filter results for {using_cache_count} ideas", context
                )

    except Exception as e:
        logger.warning(f"Error doing text diff from previous run: {e}")

    for profile in todo_profiles:
        tasks.append(
            run_profile_match(
                stocks=args.stocks,
                profile=profile,
                texts=split_texts,
                profile_filter_main_prompt_str=args.profile_filter_main_prompt,
                profile_filter_sys_prompt_str=profile_filter_sys_prompt,
                profile_output_instruction_str=args.profile_output_instruction,
                context=context,
                use_cache=False,
                detailed_log=False,
                crash_on_empty=False,
            )
        )

    filtered_stocks_list: List[List[StockID]] = await gather_with_concurrency(tasks, n=10)

    # No need to log any of this logic, logs in run_profile_match are sufficient
    stock_groups: List[StockGroup] = []
    for profile, filtered_stocks in zip(cached_profiles + todo_profiles, filtered_stocks_list):
        if profile.initial_idea in cached_filtered_stocks:
            stock_groups.append(
                StockGroup(
                    name=profile.topic,
                    stocks=filtered_stocks + cached_filtered_stocks[profile.initial_idea],
                )
            )
        elif filtered_stocks:
            stock_groups.append(StockGroup(name=profile.topic, stocks=filtered_stocks))

    return StockGroups(stock_groups=stock_groups)


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
