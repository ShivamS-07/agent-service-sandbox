import copy
import inspect
import json
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from gbi_common_py_utils.utils.pagerduty import PD_WARNING, notify_agent_pg

from agent_service.GPT.constants import GPT4_O_MINI, SONNET
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import (
    Citation,
    HistoryEntry,
    Score,
    dump_io_type,
    load_io_type,
)
from agent_service.io_types.idea import Idea
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.stock_groups import StockGroup, StockGroups
from agent_service.io_types.text import (
    StockText,
    Text,
    TextCitation,
    TextGroup,
    TextIDType,
    TopicProfiles,
)
from agent_service.planner.errors import EmptyInputError, EmptyOutputError
from agent_service.tool import (
    TOOL_DEBUG_INFO,
    ToolArgMetadata,
    ToolArgs,
    ToolCategory,
    tool,
)
from agent_service.tools.ideas.utils import ideas_enabled
from agent_service.tools.LLM_analysis.prompts import CITATION_PROMPT, CITATION_REMINDER
from agent_service.tools.LLM_analysis.utils import initial_filter_texts
from agent_service.tools.stock_filter_and_rank.constants import (
    MIN_STOCKS_FOR_RANKING,
    NONRELEVANT_COMPANY_EXPLANATION,
)
from agent_service.tools.stock_filter_and_rank.prompts import (
    COMPLEX_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT,
    FILTER_AND_RANK_STOCKS_BY_PROFILE_DESCRIPTION,
    FILTER_BY_PROFILE_DESCRIPTION,
    PER_IDEA_FILTER_AND_RANK_STOCKS_BY_PROFILE_MATCH_DESCRIPTION,
    PER_TOPIC_FILTER_BY_PROFILE_DESCRIPTION,
    PROFILE_FILTER_MAIN_PROMPT_STR_DEFAULT,
    PROFILE_OUTPUT_INSTRUCTIONS_DEFAULT,
    RANK_STOCKS_BY_PROFILE_DESCRIPTION,
    SIMPLE_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT,
)
from agent_service.tools.stock_filter_and_rank.utils import (
    CheckOutput,
    ProfileMatchParameters,
    apply_ranking_parameters,
    apply_score_threshold,
    check_text_diff,
    classify_stock_text_relevancies_for_profile,
    dedup_stocks,
    discuss_profile_fit_for_stocks,
    do_rewrite,
    evaluate_profile_fit_for_stocks,
    finalize_updates,
    get_profile_rubric,
    profile_filter_stock_match,
    rank_individual_levels,
    remove_extra_history,
    reset_relevance_cache_ids,
    stocks_rubric_score_assignment,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils import environment
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.text_utils import partition_to_smaller_text_sizes
from agent_service.utils.tool_diff import (
    get_prev_run_info,
    get_stock_text_lookup,
    get_text_diff,
)


async def run_profile_match(
    stocks: List[StockID],
    profile: Union[str, TopicProfiles],
    texts: List[StockText],
    profile_filter_main_prompt_str: str,
    profile_filter_sys_prompt_str: str,
    profile_output_instruction_str: str,
    profile_match_parameters: ProfileMatchParameters,
    context: PlanRunContext,
    profile_rubric: Optional[Dict[int, str]] = None,
    do_tool_log: bool = True,
    detailed_log: bool = True,
    crash_on_empty: bool = True,
    debug_info: Optional[Dict[str, Any]] = None,
    text_relevance_cache: Optional[List[Tuple[StockText, bool]]] = None,
    text_cache: Optional[Dict[TextIDType, str]] = None,
    no_filter: bool = False,
) -> List[StockID]:
    if context.task_id is None:
        return []  # for mypy

    if profile_match_parameters.rank_stocks:
        log_verb = "Ranking"
    else:
        log_verb = "Filtering"

    profile_str: str = ""
    if isinstance(profile, TopicProfiles):
        is_using_complex_profile = True
        profile_str = await Text.get_all_strs(  # type: ignore
            profile, include_header=False, text_group_numbering=False, text_cache=text_cache
        )
        if do_tool_log:
            await tool_log(
                f"{log_verb} stocks for advanced profile with topic: {profile.topic}",
                context=context,
            )
    elif isinstance(profile, str):
        is_using_complex_profile = False
        profile_str = profile
        if do_tool_log:
            await tool_log(f"{log_verb} stocks for simple profile: {profile_str}", context=context)
    else:
        raise ValueError(
            "profile must be either a string or a TopicProfiles object "
            "in filter_stocks_by_profile_match function!"
        )

    stocks = dedup_stocks(stocks)

    texts = await partition_to_smaller_text_sizes(texts, context=context)  # type: ignore
    split_texts_set = set(texts)

    if text_relevance_cache is not None:
        text_relevance_dict = dict(text_relevance_cache)
        new_texts = [text for text in texts if text not in text_relevance_dict]
        filtered_down_texts_tmp = await classify_stock_text_relevancies_for_profile(
            new_texts,  # type: ignore
            profiles_str=profile_str,
            context=context,
            text_cache=text_cache,
        )
        filtered_down_text_set = set(split_texts_set)
        text_relevance_cache += [
            (text, text in filtered_down_text_set)  # type: ignore
            for text in new_texts
        ]
        for text in texts:
            if text in text_relevance_dict and text_relevance_dict[text]:  # type: ignore
                filtered_down_texts_tmp.append(text)  # type: ignore
    else:
        filtered_down_texts_tmp = await classify_stock_text_relevancies_for_profile(
            texts,  # type: ignore
            profiles_str=profile_str,
            context=context,
            text_cache=text_cache,  # type: ignore
        )

        filtered_down_text_set = set(filtered_down_texts_tmp)
        text_relevance_cache = [(text, text in filtered_down_text_set) for text in texts]  # type: ignore

    filtered_down_texts: List[StockText] = [
        text for text in filtered_down_texts_tmp if text.stock_id is not None
    ]
    stocks_with_texts: List[StockID] = []
    gbi_ids_with_texts = set([text.stock_id.gbi_id for text in filtered_down_texts])  # type: ignore
    for stock in stocks:
        if stock.gbi_id in gbi_ids_with_texts:
            stocks_with_texts.append(stock)

    if do_tool_log and detailed_log:
        no_info_stock_count = len(stocks) - len(stocks_with_texts)
        if no_info_stock_count > 0:
            await tool_log(
                f"No relevant information for {no_info_stock_count} stocks, skipping these stocks",
                context=context,
            )

    aligned_text_groups = StockAlignedTextGroups.from_stocks_and_text(
        stocks_with_texts, filtered_down_texts
    )

    str_lookup: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val,
        include_header=True,
        text_group_numbering=True,
        text_cache=text_cache,
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
    if do_tool_log and detailed_log:
        await tool_log(
            f"Starting {log_verb.lower()} with {len(aligned_text_groups.val.keys())} stocks",
            context=context,
            associated_data=list(aligned_text_groups.val.keys()),
        )

    if is_using_complex_profile:
        simple_profile_filter_sys_prompt = None
        complex_profile_filter_sys_prompt = profile_filter_sys_prompt
    else:
        simple_profile_filter_sys_prompt = profile_filter_sys_prompt
        complex_profile_filter_sys_prompt = None

    if not no_filter:
        for stock, is_relevant in zip(
            aligned_text_groups.val.keys(),
            await evaluate_profile_fit_for_stocks(
                profile=profile_str,
                aligned_text_groups=aligned_text_groups,
                str_lookup=str_lookup,
                llm=cheap_llm,
                context=context,
            ),
        ):
            if is_relevant:
                stock_whitelist.add(stock)

        if do_tool_log and detailed_log:
            await tool_log(
                f"Completed a surface level round of filtering. {len(stock_whitelist)} stock(s) remaining.",
                context=context,
                associated_data=list(stock_whitelist),
            )
    llm = GPT(context=gpt_context, model=SONNET)
    stock_reason_map: Dict[StockID, Tuple[str, List[Citation]]] = {
        stock: (reason, citations)
        for stock, (is_relevant, reason, citations) in zip(
            aligned_text_groups.val.keys(),
            await profile_filter_stock_match(
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

    if stock_whitelist and do_tool_log and detailed_log:
        await tool_log(
            f"Completed a more in-depth round of filtering. {len(filtered_stocks)} stock(s) remaining.",
            context=context,
            associated_data=list(filtered_stocks),
        )

    filtered_stocks_set = set(filtered_stocks)

    needed_stocks = max(
        MIN_STOCKS_FOR_RANKING,
        profile_match_parameters.top_n if profile_match_parameters.top_n else 0,
        profile_match_parameters.bottom_m if profile_match_parameters.bottom_m else 0,
    )

    if profile_match_parameters.rank_stocks and len(filtered_stocks) < needed_stocks:
        if len(stock_whitelist) > needed_stocks:
            to_include_stocks = [
                stock for stock in stock_whitelist if stock not in filtered_stocks_set
            ]
        else:
            to_include_stocks = [
                stock for stock in stocks_with_texts if stock not in filtered_stocks_set
            ]

        if to_include_stocks:
            extra_stock_reason_map = await discuss_profile_fit_for_stocks(
                aligned_text_groups,
                str_lookup,
                profile_str,
                is_using_complex_profile,
                llm=llm,
                stock_whitelist=set(to_include_stocks),
                context=context,
            )
            if not no_filter:
                await tool_log(
                    f"Added {len(extra_stock_reason_map)} stock(s) back to list for ranking due to low filtered counts",
                    context=context,
                )
            stock_reason_map.update(extra_stock_reason_map)
            filtered_stocks_set.update(extra_stock_reason_map)
            filtered_stocks = list(filtered_stocks_set)

    # Add all the filtered out stocks back in to populate their history entries
    all_stocks = filtered_stocks
    for stock in stocks:
        if stock not in filtered_stocks_set:
            stock_reason_map[stock] = NONRELEVANT_COMPANY_EXPLANATION, []
            all_stocks.append(stock)

    # No need for an else since we can guarantee at this point one is not None, appeases linter
    if isinstance(profile, TopicProfiles):
        # TODO: Update the rubric to handle the new extensive profile data we have, for now
        # we just pass in the topic which is a short simple string similar to a profile string
        profile_data_for_rubric = profile.topic
    elif isinstance(profile, str):
        profile_data_for_rubric = profile
    else:
        raise Exception(f"logic error: profile must be a str or TopicProfile not: {type(profile)}")

    if not profile_rubric:
        profile_rubric = await get_profile_rubric(profile_data_for_rubric, context.agent_id)

    # Assigns scores inplace
    filtered_stocks_with_scores = await stocks_rubric_score_assignment(
        all_stocks,
        profile_rubric,
        stock_reason_map,
        profile_data_for_rubric,
        context,
    )

    final_stocks = filtered_stocks_with_scores

    # finally, we do fine-grained ranking of those stocks which need it, if ranking
    if profile_match_parameters.rank_stocks and final_stocks:
        final_stocks = await rank_individual_levels(
            profile=profile_str,
            stocks=final_stocks,
            context=context,
        )

    if debug_info is not None:
        debug_info["profile_rubric"] = json.dumps(profile_rubric)
        debug_info["text_relevance_cache"] = dump_io_type(
            remove_extra_history(text_relevance_cache)
        )
        debug_info["full_stock_list"] = dump_io_type(final_stocks)

    final_stocks = apply_score_threshold(
        final_stocks, profile_match_parameters.filter_score_threshold
    )

    if not final_stocks and crash_on_empty:
        raise EmptyOutputError(
            message=f"Stock profile filter looking for '{profile_data_for_rubric}' resulted in an empty list of stocks"
        )

    if profile_match_parameters.rank_stocks:
        final_stocks = await apply_ranking_parameters(
            ranked_stocks=final_stocks,
            profile=profile,
            context=context,
            top_n=profile_match_parameters.top_n,
            bottom_m=profile_match_parameters.bottom_m,
        )

    final_stocks.sort(key=lambda stock: stock.history[-1].score.val, reverse=True)  # type: ignore
    if do_tool_log:
        if profile_match_parameters.rank_stocks:
            await tool_log(
                f"A total of {len(final_stocks)} stocks ranked for '{profile_data_for_rubric}'",
                context,
            )
        else:
            await tool_log(
                f"A total of {len(final_stocks)} stocks passed the filter for '{profile_data_for_rubric}'",
                context,
            )

    return final_stocks


async def update_profile_match(
    stocks: List[StockID],
    prev_input_stocks: List[StockID],
    prev_output_stocks: List[StockID],
    profile: Union[str, TopicProfiles],
    profile_rubric: Dict[int, str],
    texts: List[StockText],
    prev_texts: List[StockText],
    profile_filter_main_prompt_str: str,
    profile_filter_sys_prompt_str: str,
    profile_output_instruction_str: str,
    profile_match_parameters: ProfileMatchParameters,
    context: PlanRunContext,
    detailed_log: bool = True,
    crash_on_empty: bool = True,
    text_relevance_cache: Optional[List[Tuple[StockText, bool]]] = None,
    debug_info: Optional[Dict[str, Any]] = None,
    text_cache: Optional[Dict[TextIDType, str]] = None,
) -> List[StockID]:
    stocks = dedup_stocks(stocks)
    prev_input_stocks = dedup_stocks(stocks)

    final_rank_changed_stocks = []
    rank_unchanged_stocks = []
    added_stocks = {}
    modified_stocks = {}
    removed_stocks = {}

    if isinstance(profile, TopicProfiles):
        profile_data_for_rubric = profile.topic
    elif isinstance(profile, str):
        profile_data_for_rubric = profile
    else:
        raise Exception(f"logic error: profile must be a str or TopicProfile not: {type(profile)}")

    if not profile_rubric:
        profile_rubric = await get_profile_rubric(profile_data_for_rubric, context.agent_id)

    profile_str: str = ""
    if isinstance(profile, TopicProfiles):
        is_using_complex_profile = True
        profile_str = await Text.get_all_strs(  # type: ignore
            profile, include_header=False, text_group_numbering=False, text_cache=text_cache
        )
    elif isinstance(profile, str):
        is_using_complex_profile = False
        profile_str = profile
    else:
        raise ValueError(
            "profile must be either a string or a TopicProfiles object "
            "in filter_stocks_by_profile_match function!"
        )

    # first, do check for new stocks, run them through regular main pipeline if any
    old_stocks = set(prev_input_stocks)
    new_stocks = [stock for stock in stocks if stock not in old_stocks]
    if new_stocks:
        if detailed_log:
            await tool_log(log=f"Processing {len(new_stocks)} new stocks", context=context)
        sub_match_parameters = copy.copy(profile_match_parameters)
        sub_match_parameters.rank_stocks = False

        temp_debug: Dict[str, str] = {}
        relevant_gbi_ids = set([stock.gbi_id for stock in new_stocks])
        new_stock_texts = [
            text for text in texts if text.stock_id and text.stock_id.gbi_id in relevant_gbi_ids
        ]
        texts = [
            text for text in texts if text.stock_id and text.stock_id.gbi_id not in relevant_gbi_ids
        ]
        stocks = [stock for stock in stocks if stock.gbi_id not in relevant_gbi_ids]
        final_rank_changed_stocks.extend(
            await run_profile_match(
                stocks=new_stocks,
                profile=profile,
                texts=new_stock_texts,
                profile_rubric=profile_rubric,
                profile_filter_main_prompt_str=profile_filter_main_prompt_str,
                profile_filter_sys_prompt_str=profile_filter_sys_prompt_str,
                profile_output_instruction_str=profile_output_instruction_str,
                profile_match_parameters=sub_match_parameters,
                context=context,
                detailed_log=False,
                crash_on_empty=False,
                debug_info=temp_debug,
            )
        )
        new_stock_relevance_cache = temp_debug["text_relevance_cache"]
    else:
        new_stock_relevance_cache = None

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=SONNET)

    # Next, look for new texts, and check stocks with new texts for relevant check

    split_texts = await partition_to_smaller_text_sizes(texts, context=context)  # type: ignore
    split_text_lookup = get_stock_text_lookup(split_texts)  # type: ignore

    spilt_texts_set = set(split_texts)

    # this removes outdated text from the cache so we aren't storing them forever
    if text_relevance_cache:
        text_relevance_cache = [pair for pair in text_relevance_cache if pair[0] in spilt_texts_set]
    else:
        text_relevance_cache = []

    prev_stock_text_lookup = get_stock_text_lookup(prev_texts)
    current_stock_text_lookup = get_stock_text_lookup(texts)

    check_tasks = []
    prev_output_lookup = {stock.gbi_id: stock for stock in prev_output_stocks}
    no_major_new_texts_stocks = []

    for stock in stocks:
        text_diff = get_text_diff(
            current_stock_text_lookup.get(stock, []), prev_stock_text_lookup.get(stock, [])
        )
        if text_diff:
            text_diff_id_set = set([text.id for text in text_diff])
            new_split_texts = [
                text
                for text in split_text_lookup.get(stock, [])
                if text.get_original_text_id() in text_diff_id_set
            ]
            check_tasks.append(
                check_text_diff(
                    stock,
                    new_split_texts,
                    profile_str,
                    profile_rubric,
                    prev_output_lookup.get(stock.gbi_id),
                    llm,
                    context,
                    text_cache=text_cache,
                )
            )
        else:
            no_major_new_texts_stocks.append(stock)

    check_results = await gather_with_concurrency(check_tasks)

    rewrite_results = []

    for check_result in check_results:
        text_relevance_cache.extend(check_result.text_relevance_cache)
        if check_result.changed:
            if check_result.new_score != 0:
                if check_result.prev_score == 0:
                    added_stocks[check_result.stock] = check_result
                else:
                    modified_stocks[check_result.stock] = check_result
                rewrite_results.append(check_result)
            else:
                removed_stocks[check_result.stock] = check_result
                final_rank_changed_stocks.append(
                    check_result.stock.inject_history_entry(
                        HistoryEntry(
                            explanation=check_result.explanation,
                            citations=check_result.citations,
                            score=Score(val=0.0),
                        )
                    )
                )

        else:
            no_major_new_texts_stocks.append(check_result.stock)

    # this makes sure the text relevance cache is fully up to date

    text_relevance_lookup = dict(text_relevance_cache)

    rewrite_tasks = []

    to_check_texts = []
    for stock in no_major_new_texts_stocks + [
        check_result.stock for check_result in rewrite_results
    ]:
        for text in split_text_lookup.get(stock, []):
            if text not in text_relevance_lookup:
                to_check_texts.append(text)

    passed_texts = await classify_stock_text_relevancies_for_profile(
        to_check_texts,
        profiles_str=profile_str,
        context=context,  # type: ignore
    )
    passed_texts_set = set(passed_texts)
    for text in to_check_texts:
        if text in passed_texts_set:
            text_relevance_lookup[text] = True
            text_relevance_cache.append((text, True))
        else:
            text_relevance_lookup[text] = False
            text_relevance_cache.append((text, False))

    # Now we rewrite those texts which had new data

    if rewrite_results and detailed_log:
        await tool_log(
            f"Updating {len(rewrite_results) + len(removed_stocks)} stock(s) due to new relevant information",
            context,
        )

    for check_result in rewrite_results:
        relevant_texts = [
            text
            for text in split_text_lookup.get(check_result.stock, [])
            if text_relevance_lookup[text]
        ]
        rewrite_tasks.append(
            do_rewrite(
                check_result,
                relevant_texts,
                is_using_complex_profile,
                profile_str,
                profile_data_for_rubric,
                llm,
                context,
            )
        )

    final_rank_changed_stocks.extend(await gather_with_concurrency(rewrite_tasks))

    # Now we look through through existing outputs for missing citations, and rewrite if needed

    text_group_dict = {}

    removed = 0
    old_output = None

    for stock in no_major_new_texts_stocks:
        if stock.gbi_id in prev_output_lookup:
            old_output = prev_output_lookup[stock.gbi_id]
            missing_citations = False
            for citation in old_output.history[-1].citations:
                if isinstance(citation, TextCitation):
                    citation.source_text.reset_id()
                    if citation.source_text not in text_relevance_lookup:
                        missing_citations = True
                        break
            if missing_citations:
                relevant_texts = [
                    text for text in split_text_lookup.get(stock, []) if text_relevance_lookup[text]
                ]
                if relevant_texts:
                    text_group_dict[stock] = TextGroup(val=relevant_texts)  # type: ignore
                    continue
                else:
                    stock_with_history = stock.inject_history_entry(
                        HistoryEntry(
                            explanation=NONRELEVANT_COMPANY_EXPLANATION,
                            citations=[],
                            score=Score(val=0.0),
                        )
                    )
                    removed_stocks[stock] = CheckOutput(
                        stock,
                        old_output,
                        True,
                        explanation=NONRELEVANT_COMPANY_EXPLANATION,
                        citations=[],
                        new_score=0,
                    )
                    removed += 1
            else:
                stock_with_history = stock.inject_history_entry(old_output.history[-1])
        else:
            stock_with_history = stock.inject_history_entry(
                HistoryEntry(
                    explanation=NONRELEVANT_COMPANY_EXPLANATION, citations=[], score=Score(val=0.0)
                )
            )
        rank_unchanged_stocks.append(stock_with_history)

    if text_group_dict and detailed_log:
        await tool_log(
            f"Updating {len(text_group_dict) + removed} stock(s) due to cited texts not in time window",
            context,
        )

    aligned_text_groups = StockAlignedTextGroups(val=text_group_dict)

    str_lookup: Dict[StockID, str] = await Text.get_all_strs(  # type: ignore
        aligned_text_groups.val,
        include_header=True,
        text_group_numbering=True,
        text_cache=text_cache,
    )

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

    if is_using_complex_profile:
        simple_profile_filter_sys_prompt = None
        complex_profile_filter_sys_prompt = profile_filter_sys_prompt
    else:
        simple_profile_filter_sys_prompt = profile_filter_sys_prompt
        complex_profile_filter_sys_prompt = None

    stock_reason_map: Dict[StockID, Tuple[str, List[Citation]]] = {
        stock: (reason, citations)
        for stock, (is_relevant, reason, citations) in zip(
            aligned_text_groups.val.keys(),
            await profile_filter_stock_match(
                aligned_text_groups,
                str_lookup,
                profile_str,
                isinstance(profile, TopicProfiles),
                llm=llm,
                profile_filter_main_prompt=profile_filter_main_prompt,
                simple_profile_filter_sys_prompt=simple_profile_filter_sys_prompt,
                complex_profile_filter_sys_prompt=complex_profile_filter_sys_prompt,
                context=context,
                do_citations=True,
            ),
        )
        if is_relevant
    }

    for stock in text_group_dict:
        output_stock_history = prev_output_lookup[stock.gbi_id].history[-1]
        if stock in stock_reason_map:
            output_stock_history.explanation = stock_reason_map[stock][0]
            output_stock_history.citations = stock_reason_map[stock][1]
            # we don't change the score here because we only change the score
            # when there was a major new text
        else:
            if output_stock_history.score and output_stock_history.score.val != 0:
                removed_stocks[stock] = CheckOutput(
                    stock,
                    old_output,
                    True,
                    explanation=NONRELEVANT_COMPANY_EXPLANATION,
                    citations=[],
                    new_score=0,
                )
            output_stock_history.score = Score(val=0)
            output_stock_history.explanation = NONRELEVANT_COMPANY_EXPLANATION
            output_stock_history.citations = []
        rank_unchanged_stocks.append(stock.inject_history_entry(output_stock_history))

    # finally, we do fine-grained ranking of those stocks which need it, if ranking
    if profile_match_parameters.rank_stocks and final_rank_changed_stocks:
        if text_group_dict and detailed_log:
            await tool_log("Updating fine-grained ranking", context)
        final_stocks = await rank_individual_levels(
            profile=profile_str,
            stocks=final_rank_changed_stocks,
            fixed_stocks=rank_unchanged_stocks,
            context=context,
        )
    else:
        final_stocks = final_rank_changed_stocks + rank_unchanged_stocks

    if new_stock_relevance_cache:
        text_relevance_cache += cast(
            List[Tuple[StockText, bool]], load_io_type(new_stock_relevance_cache)
        )

    if debug_info is not None:
        debug_info["full_stock_list"] = dump_io_type(final_stocks)
        debug_info["profile_rubric"] = json.dumps(profile_rubric)
        debug_info["text_relevance_cache"] = dump_io_type(
            remove_extra_history(text_relevance_cache)
        )

    final_stocks = apply_score_threshold(
        final_stocks, profile_match_parameters.filter_score_threshold
    )

    # Once final_stocks have been saved, we trim it down to the top_n, bottom_m if applicable
    if profile_match_parameters.rank_stocks:
        final_stocks = await apply_ranking_parameters(
            ranked_stocks=final_stocks,
            profile=profile,
            context=context,
            top_n=profile_match_parameters.top_n,
            bottom_m=profile_match_parameters.bottom_m,
        )

    if crash_on_empty and not final_stocks:
        raise EmptyOutputError(
            message=f"Stock profile filter looking for '{profile_str}' resulted in an empty list of stocks"
        )

    if context.diff_info is not None and context.task_id:
        context.diff_info[context.task_id] = {}
        final_output_lookup = {stock_id: stock_id for stock_id in final_stocks}
        context.diff_info[context.task_id]["added"] = finalize_updates(
            added_stocks, final_output_lookup, profile_match_parameters.rank_stocks
        )
        context.diff_info[context.task_id]["removed"] = finalize_updates(
            removed_stocks, final_output_lookup, profile_match_parameters.rank_stocks
        )
        context.diff_info[context.task_id]["modified"] = finalize_updates(
            modified_stocks, final_output_lookup, profile_match_parameters.rank_stocks
        )

    final_stocks.sort(key=lambda stock: stock.history[-1].score.val, reverse=True)  # type: ignore
    if profile_match_parameters.rank_stocks:
        await tool_log(
            f"A total of {len(final_stocks)} stocks ranked for '{profile_data_for_rubric}'",
            context,
        )
    else:
        await tool_log(
            f"A total of {len(final_stocks)} stocks passed the filter for '{profile_data_for_rubric}'",
            context,
        )

    return final_stocks


class FilterStocksByProfileMatch(ToolArgs):
    stocks: List[StockID]
    texts: List[StockText]
    profile: Union[str, TopicProfiles]


@tool(
    description=FILTER_BY_PROFILE_DESCRIPTION,
    category=ToolCategory.STOCK_FILTERS,
    enabled=False,
)
async def filter_stocks_by_profile_match(
    args: FilterStocksByProfileMatch, context: PlanRunContext
) -> List[StockID]:
    return await filter_and_rank_stocks_by_profile(
        FilterAndRankStocksByProfileInput(
            stocks=args.stocks,
            stock_texts=args.texts,
            profile=args.profile,
            complete_ranking=False,
            score_threshold=0,
            caller_func=filter_stocks_by_profile_match.__name__,
        ),
        context=context,
    )  # type: ignore


class RankStocksByProfileInput(ToolArgs):
    stocks: List[StockID]
    stock_texts: List[StockText]
    profile: Union[str, TopicProfiles]
    top_n: Optional[int] = None
    bottom_m: Optional[int] = None


@tool(
    description=RANK_STOCKS_BY_PROFILE_DESCRIPTION,
    category=ToolCategory.STOCK_FILTERS,
    enabled=False,
)
async def rank_stocks_by_profile(
    args: RankStocksByProfileInput, context: PlanRunContext
) -> List[StockID]:
    return await filter_and_rank_stocks_by_profile(
        FilterAndRankStocksByProfileInput(
            stocks=args.stocks,
            stock_texts=args.stock_texts,
            profile=args.profile,
            complete_ranking=True,
            score_threshold=0,
            top_n=args.top_n,
            bottom_m=args.bottom_m,
            caller_func=rank_stocks_by_profile.__name__,
        ),
        context=context,
    )  # type: ignore


class FilterAndRankStocksByProfileInput(ToolArgs):
    stocks: List[StockID]
    stock_texts: List[StockText]
    profile: Union[str, TopicProfiles]
    complete_ranking: bool
    score_threshold: int = 0
    top_n: Optional[int] = None
    bottom_m: Optional[int] = None
    no_filter: bool = False

    # prompt arguments (hidden from planner)
    profile_filter_main_prompt: str = PROFILE_FILTER_MAIN_PROMPT_STR_DEFAULT
    simple_profile_filter_sys_prompt: str = SIMPLE_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT
    complex_profile_filter_sys_prompt: str = COMPLEX_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT
    profile_output_instruction: str = PROFILE_OUTPUT_INSTRUCTIONS_DEFAULT
    caller_func: Optional[str] = None

    # tool arguments metadata
    arg_metadata = {
        "profile_filter_main_prompt": ToolArgMetadata(hidden_from_planner=True),
        "simple_profile_filter_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
        "complex_profile_filter_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
        "profile_output_instruction": ToolArgMetadata(hidden_from_planner=True),
        "caller_func": ToolArgMetadata(hidden_from_planner=True),
    }


@tool(
    description=FILTER_AND_RANK_STOCKS_BY_PROFILE_DESCRIPTION,
    category=ToolCategory.STOCK_FILTERS,
)
async def filter_and_rank_stocks_by_profile(
    args: FilterAndRankStocksByProfileInput, context: PlanRunContext
) -> List[StockID]:
    logger = get_prefect_logger(__name__)

    if len(args.stocks) == 0:
        raise EmptyInputError("Cannot run on an empty list of stocks")
    if len(args.stock_texts) == 0:
        raise EmptyInputError("Cannot run on stocks with an empty list of texts")

    profile_match_params = ProfileMatchParameters(
        filter_score_threshold=args.score_threshold,
        rank_stocks=args.complete_ranking,
        top_n=args.top_n,
        bottom_m=args.bottom_m,
    )

    caller_input_dataclass = None
    # this is only needed for efficient updates when switching, can be removed later
    if args.caller_func:
        caller_func_name = args.caller_func
        if caller_func_name == "rank_stocks_by_profile":
            caller_input_dataclass = RankStocksByProfileInput
        elif caller_func_name == "filter_stocks_by_profile_match":
            caller_input_dataclass = FilterStocksByProfileMatch  # type: ignore
    else:
        caller_func_name = None

    profile_filter_sys_prompt = ""
    if isinstance(args.profile, str):
        profile_filter_sys_prompt = args.simple_profile_filter_sys_prompt
    elif isinstance(args.profile, TopicProfiles):
        profile_filter_sys_prompt = args.complex_profile_filter_sys_prompt
    else:
        raise ValueError(
            "profile must be either a string or a TopicProfiles object "
            "in filter_and_rank_by_profile function!"
        )

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    result = None
    prev_stocks = []
    prev_stock_texts = []
    text_relevance_cache = []
    full_output: List[StockID] = []
    profile_rubric: Dict[int, str] = {}

    text_cache: Dict[TextIDType, str] = {}
    # Pre-fetch texts so the cache is populated
    _ = await Text.get_all_strs(args.stock_texts, text_cache=text_cache)

    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(context, "filter_and_rank_stocks_by_profile")
        if prev_run_info is not None:
            prev_args = FilterAndRankStocksByProfileInput.model_validate_json(
                prev_run_info.inputs_str
            )
            prev_other = prev_run_info.debug
            full_output = cast(List[StockID], load_io_type(prev_other["full_stock_list"]))
            profile_rubric = cast(Dict[int, str], load_io_type(prev_other["profile_rubric"]))
            text_relevance_cache = cast(
                List[Tuple[StockText, bool]], load_io_type(prev_other["text_relevance_cache"])
            )
            text_relevance_cache = reset_relevance_cache_ids(text_relevance_cache)
            prev_stocks = args.stocks
            prev_stock_texts = prev_args.stock_texts
        elif caller_func_name:  # this is for efficient update switchover, can be removed later
            prev_run_info = await get_prev_run_info(context, caller_func_name)
            if prev_run_info is not None:
                prev_args = caller_input_dataclass.model_validate_json(  # type:ignore
                    prev_run_info.inputs_str
                )

                full_output = cast(List[StockID], prev_run_info.output)
                profile_rubric = {}
                text_relevance_cache = []
                prev_stocks = args.stocks
                if caller_func_name == "filter_stocks_by_profile_match":
                    prev_stock_texts = prev_args.texts  # type: ignore
                elif caller_func_name == "rank_stocks_by_profile":
                    prev_stock_texts = prev_args.stock_texts

                # Pre-fetch texts so the cache is populated
                _ = await Text.get_all_strs(prev_stock_texts, text_cache=text_cache)

        if prev_run_info is not None:
            result = await update_profile_match(
                stocks=args.stocks,
                prev_input_stocks=prev_stocks,
                prev_output_stocks=full_output,
                profile=args.profile,
                profile_rubric=profile_rubric,
                texts=args.stock_texts,
                prev_texts=prev_stock_texts,
                profile_filter_main_prompt_str=args.profile_filter_main_prompt,
                profile_filter_sys_prompt_str=profile_filter_sys_prompt,
                profile_output_instruction_str=args.profile_output_instruction,
                profile_match_parameters=profile_match_params,
                context=context,
                text_relevance_cache=text_relevance_cache,
                debug_info=debug_info,
                text_cache=text_cache,
            )

    except EmptyOutputError as e:
        raise e

    except Exception as e:
        logger.exception(f"Error doing text diff from previous run: {e}")
        # some of the exception messages are long and contain unique info,
        # truncate it to try to prevent that from making too many unique pagers
        error_dedupe_str = str(type(e)) + " " + str(e)[:75]
        func_name = __name__
        frame = inspect.currentframe()
        if frame:
            # defined to be potentially null, in practice it never is
            func_name = frame.f_code.co_name
        classt = "AgentUpdateError"
        group = f"{classt}-{func_name}-{error_dedupe_str}"
        notify_agent_pg(
            summary=f"{func_name}: Failed to update per stock summary: {error_dedupe_str}",
            severity=PD_WARNING,
            source=environment.get_environment_tag(),
            component="AgentError",
            classt=classt,
            group=group,
            custom_details={
                "_reminder": "This pager is deduped, check #oncall-info for more examples",
                "agent": context.agent_id,
                "plan_run": context.plan_run_id,
                "task": context.task_id,
                "error": "".join(traceback.TracebackException.from_exception(e).format()),
                "pagerduty_dedupe_key": group,
            },
        )

    if result is None:
        result = await run_profile_match(
            args.stocks,
            args.profile,
            args.stock_texts,
            profile_filter_main_prompt_str=args.profile_filter_main_prompt,
            profile_filter_sys_prompt_str=profile_filter_sys_prompt,
            profile_output_instruction_str=args.profile_output_instruction,
            profile_match_parameters=profile_match_params,
            context=context,
            debug_info=debug_info,
            no_filter=args.no_filter,
        )
    return result or []


class PerIdeaFilterStocksByProfileMatch(ToolArgs):
    stocks: List[StockID]
    texts: List[StockText]
    ideas: List[Idea]
    profiles: List[TopicProfiles]


@tool(
    description=PER_TOPIC_FILTER_BY_PROFILE_DESCRIPTION,
    category=ToolCategory.STOCK_FILTERS,
    enabled=False,
    enabled_checker_func=ideas_enabled,
)
async def per_idea_filter_stocks_by_profile_match(
    args: PerIdeaFilterStocksByProfileMatch,
    context: PlanRunContext,
) -> StockGroups:
    return await per_idea_filter_and_rank_stocks_by_profile_match(
        PerIdeaFilterAndRankStocksByProfileMatch(
            stocks=args.stocks,
            stock_texts=args.texts,
            ideas=args.ideas,
            profiles=args.profiles,
            complete_ranking=False,
            score_threshold=0,
        ),
        context=context,
    )  # type: ignore


class PerIdeaFilterAndRankStocksByProfileMatch(ToolArgs):
    stocks: List[StockID]
    stock_texts: List[StockText]
    ideas: List[Idea]
    profiles: List[TopicProfiles]
    complete_ranking: bool
    score_threshold: int = 0
    top_n: Optional[int] = None
    bottom_m: Optional[int] = None

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
        "overridden_caller_func": ToolArgMetadata(hidden_from_planner=True),
    }


@tool(
    description=PER_IDEA_FILTER_AND_RANK_STOCKS_BY_PROFILE_MATCH_DESCRIPTION,
    category=ToolCategory.IDEAS,
    enabled=True,
    enabled_checker_func=ideas_enabled,
)
async def per_idea_filter_and_rank_stocks_by_profile_match(
    args: PerIdeaFilterAndRankStocksByProfileMatch,
    context: PlanRunContext,
) -> StockGroups:
    logger = get_prefect_logger(__name__)
    if len(args.stocks) == 0:
        raise EmptyInputError("Cannot run this operation on an empty list of stocks")
    if len(args.stock_texts) == 0:
        raise EmptyInputError("Cannot run this operation on an empty list of texts")
    if len(args.profiles) == 0:
        raise EmptyInputError("Cannot run this operation with an empty list of profiles")
    if len(args.ideas) == 0:
        raise EmptyInputError("Cannot run this operation with an empty list of ideas")

    profile_filter_sys_prompt = args.complex_profile_filter_sys_prompt

    profile_idea_lookup: Dict[str, Idea] = {}
    for profile in args.profiles:
        for idea in args.ideas:
            if profile.initial_idea == idea.title:
                profile_idea_lookup[profile.initial_idea] = idea

    prev_run_info = None

    debug_dicts: Dict[str, Dict[str, str]] = {
        profile.initial_idea: {} for profile in args.profiles if profile.initial_idea
    }

    todo_profiles = args.profiles[:]

    removed_stocks: Set[StockID] = set()
    cached_profiles: List[TopicProfiles] = []
    cached_filtered_stocks = {}

    # TODO: doing this both here and inside run_profile_match is redundant, but basically has no
    # effect, and useful to have them available for checking against citations
    split_texts = cast(
        List[StockText],
        await partition_to_smaller_text_sizes(args.stock_texts, context=context),  # type: ignore
    )

    text_cache: Dict[TextIDType, str] = {}
    # Pre-fetch texts so the cache is populated
    _ = await Text.get_all_strs(split_texts, text_cache=text_cache)

    tasks = []

    # Disabled for ranking as it does not have update logic designed yet
    profile_match_parameters = ProfileMatchParameters()
    try:  # since everything associated with diffing is optional, put in try/except
        prev_run_info = await get_prev_run_info(
            context, "per_idea_filter_and_rank_stocks_by_profile_match"
        )
        if prev_run_info is not None:
            prev_args = PerIdeaFilterAndRankStocksByProfileMatch.model_validate_json(
                prev_run_info.inputs_str
            )
            old_ideas = set(prev_args.ideas)
            old_stocks = set(prev_args.stocks)
            new_stocks = set(args.stocks)
            added_stocks = new_stocks - old_stocks
            removed_stocks = old_stocks - new_stocks
            all_texts = set(args.stock_texts) | set(split_texts)
            prev_output: StockGroups = prev_run_info.output  # type:ignore
            prev_group_lookup = {group.name: group for group in prev_output.stock_groups}
            using_cache_count = 0
            for profile in args.profiles:
                if (
                    profile.initial_idea in profile_idea_lookup
                    and profile_idea_lookup[profile.initial_idea] in old_ideas
                    and profile.topic in prev_group_lookup
                ):
                    if prev_run_info.debug:
                        prev_debug_dict = json.loads(prev_run_info.debug[profile.initial_idea])
                        profile_rubric = json.loads(prev_debug_dict["profile_rubric"])
                        text_relevance_cache = cast(
                            List[Tuple[StockText, bool]],
                            load_io_type(prev_debug_dict["text_relevance_cache"]),
                        )
                        text_relevance_cache = reset_relevance_cache_ids(text_relevance_cache)
                    else:
                        profile_rubric = None
                        text_relevance_cache = []

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
                                profile_rubric=profile_rubric,
                                texts=split_texts,
                                profile_match_parameters=profile_match_parameters,
                                profile_filter_main_prompt_str=args.profile_filter_main_prompt,
                                profile_filter_sys_prompt_str=profile_filter_sys_prompt,
                                profile_output_instruction_str=args.profile_output_instruction,
                                context=context,
                                crash_on_empty=False,
                                detailed_log=False,
                                debug_info=debug_dicts[profile.initial_idea],
                                text_relevance_cache=text_relevance_cache,
                                text_cache=text_cache,
                            )
                        )
                    else:
                        tasks.append(identity([]))

                    if not must_do_stocks:
                        debug_dicts[profile.initial_idea]["profile_rubric"] = json.dumps(
                            profile_rubric
                        )
                        debug_dicts[profile.initial_idea]["text_relevance_cache"] = dump_io_type(
                            remove_extra_history(text_relevance_cache)
                        )

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

    except EmptyOutputError as e:
        raise e
    except Exception as e:
        logger.warning(f"Error doing text diff from previous run: {e}")

    initial_split_text_count = len(split_texts)
    split_texts = cast(List[StockText], initial_filter_texts(split_texts))  # type: ignore
    if len(split_texts) != initial_split_text_count:
        logger.warning(
            f"Too many texts, filtered {initial_split_text_count} split texts to {len(split_texts)}"
        )

    for profile in todo_profiles:
        profile_match_params = ProfileMatchParameters(
            filter_score_threshold=args.score_threshold,
            rank_stocks=args.complete_ranking,
            top_n=args.top_n,
            bottom_m=args.bottom_m,
        )

        tasks.append(
            run_profile_match(
                stocks=args.stocks,
                profile=profile,
                texts=split_texts,
                profile_match_parameters=profile_match_params,
                profile_filter_main_prompt_str=args.profile_filter_main_prompt,
                profile_filter_sys_prompt_str=profile_filter_sys_prompt,
                profile_output_instruction_str=args.profile_output_instruction,
                context=context,
                detailed_log=False,
                crash_on_empty=False,
                debug_info=debug_dicts[profile.initial_idea] if profile.initial_idea else {},
                text_cache=text_cache,
            )
        )

    filtered_stocks_list: List[List[StockID]] = await gather_with_concurrency(tasks, n=10)

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    for profile_str, debug_dict in debug_dicts.items():
        debug_info[profile_str] = json.dumps(debug_dict)

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
