from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from agent_service.GPT.constants import GPT4_O_MINI, SONNET
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import Citation, dump_io_type
from agent_service.io_types.idea import Idea
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.stock_groups import StockGroup, StockGroups
from agent_service.io_types.text import StockText, Text, TextCitation, TopicProfiles
from agent_service.planner.errors import EmptyInputError, EmptyOutputError
from agent_service.tool import (
    TOOL_DEBUG_INFO,
    ToolArgMetadata,
    ToolArgs,
    ToolCategory,
    tool,
)
from agent_service.tools.ideas.utils import ideas_enabled
from agent_service.tools.LLM_analysis.constants import NO_CITATIONS_DIFF
from agent_service.tools.LLM_analysis.prompts import CITATION_PROMPT, CITATION_REMINDER
from agent_service.tools.LLM_analysis.utils import (
    classify_stock_text_relevancies_for_profile,
)
from agent_service.tools.stock_filter_and_rank.constants import MAX_RUBRIC_SCORE
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
    evaluate_profile_fit_for_stocks,
    get_profile_rubric,
    profile_filter_added_diff_info,
    profile_filter_removed_diff_info,
    profile_filter_stock_match,
    rank_individual_levels,
    stocks_rubric_score_assignment,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.text_utils import partition_to_smaller_text_sizes
from agent_service.utils.tool_diff import (
    add_old_history,
    get_prev_run_info,
    get_stock_text_lookup,
    get_text_diff,
)


@dataclass
class ProfileMatchParameters:
    filter_score_threshold: int = 1
    rank_stocks: bool = False
    top_n: Optional[int] = None
    bottom_m: Optional[int] = None


async def run_profile_match(
    stocks: List[StockID],
    profile: Union[str, TopicProfiles],
    texts: List[StockText],
    caller_func_name: str,
    profile_filter_main_prompt_str: str,
    profile_filter_sys_prompt_str: str,
    profile_output_instruction_str: str,
    profile_match_parameters: ProfileMatchParameters,
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
            prev_run_info = await get_prev_run_info(context, caller_func_name)
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
            associated_data=list(aligned_text_groups.val.keys()),
        )

    if is_using_complex_profile:
        simple_profile_filter_sys_prompt = None
        complex_profile_filter_sys_prompt = profile_filter_sys_prompt
    else:
        simple_profile_filter_sys_prompt = profile_filter_sys_prompt
        complex_profile_filter_sys_prompt = None

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

    if detailed_log:
        await tool_log(
            f"Completed a surface level round of filtering. {len(stock_whitelist)} stocks remaining.",
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

    if detailed_log:
        await tool_log(
            f"Completed a more in-depth round of filtering. {len(filtered_stocks)} stocks remaining.",
            context=context,
            associated_data=list(filtered_stocks),
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
    if profile_match_parameters.filter_score_threshold != 0:
        await tool_log(
            f"Only keeping stocks with a score higher than {profile_match_parameters.filter_score_threshold}",
            context=context,
        )
    for stock in filtered_stocks_with_scores:
        if stock.company_name not in company_names:
            try:
                # Need to multiple by the max rubric score to convert from the 0-1 score to the 0-5 level system
                adjusted_score = (stock.history[-1].score.val) * MAX_RUBRIC_SCORE  # type: ignore
                if adjusted_score >= profile_match_parameters.filter_score_threshold:
                    company_names.add(stock.company_name)
                    dedup_res.append(stock)
            except ValueError:
                logger.warning(
                    f"{stock.company_name} ({stock.gbi_id}) had no score during profile match, {stock.history[-1]}"
                )
    dedup_res = sorted(
        dedup_res, key=lambda stock: stock.history[-1].score.val, reverse=True  # type: ignore
    )

    if profile_match_parameters.rank_stocks:
        logger.info("Applying inter-level ranking to individually rank all stocks...")
        fully_ranked_stocks = await rank_individual_levels(
            profile_str, dedup_res, context, top_n=profile_match_parameters.top_n
        )

        # Use a set for top_n & bottom_m so as to avoid cases where we return duplicate stocks
        # if top_n + bottom_m > len(fully_ranked_stocks)
        truncated_ranked_stocks = set()
        if profile_match_parameters.top_n:
            logger.info(f"Determined the top {profile_match_parameters.top_n}")
            top_stocks = fully_ranked_stocks[: profile_match_parameters.top_n]
            non_zero_top_stocks = [stock for stock in top_stocks if stock.history[-1].score.val != 0]  # type: ignore

            if len(non_zero_top_stocks) == 0:
                profile_topic = profile if isinstance(profile, str) else profile.topic
                await tool_log(
                    "Could not find any relavent stocks from the given set relevant "
                    f"to '{profile_topic}'",
                    context=context,
                )
            elif (len(non_zero_top_stocks) < len(top_stocks)) or (
                len(non_zero_top_stocks) < profile_match_parameters.top_n
            ):
                await tool_log(
                    f"Only able to find {len(non_zero_top_stocks)} top stocks, "
                    "all other stocks were not relevant",
                    context=context,
                )
            else:
                await tool_log(
                    f"Determined the top {profile_match_parameters.top_n}",
                    context=context,
                )
            truncated_ranked_stocks.update(non_zero_top_stocks)
        if profile_match_parameters.bottom_m:
            logger.info(f"Determined the bottom {profile_match_parameters.bottom_m}")
            await tool_log(
                f"Determined the bottom {profile_match_parameters.bottom_m}",
                context=context,
            )
            truncated_ranked_stocks.update(
                fully_ranked_stocks[profile_match_parameters.bottom_m * (-1) :]
            )
        if profile_match_parameters.top_n or profile_match_parameters.bottom_m:
            truncated_stock_list = sorted(
                list(truncated_ranked_stocks),
                key=lambda stock: stock.history[-1].score.val,  # type: ignore
                reverse=True,
            )
            return truncated_stock_list
        else:
            non_zero_ranked_stocks = [
                stock for stock in fully_ranked_stocks if stock.history[-1].score.val != 0  # type: ignore
            ]
            return non_zero_ranked_stocks
    else:
        return dedup_res


class FilterStocksByProfileMatch(ToolArgs):
    stocks: List[StockID]
    texts: List[StockText]
    profile: Union[str, TopicProfiles]


@tool(
    description=FILTER_BY_PROFILE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
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
            score_threshold=1,
            overridden_caller_func=filter_stocks_by_profile_match.__name__,
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
    category=ToolCategory.LLM_ANALYSIS,
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
            score_threshold=1,
            top_n=args.top_n,
            bottom_m=args.bottom_m,
            overridden_caller_func=rank_stocks_by_profile.__name__,
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

    # prompt arguments (hidden from planner)
    profile_filter_main_prompt: str = PROFILE_FILTER_MAIN_PROMPT_STR_DEFAULT
    simple_profile_filter_sys_prompt: str = SIMPLE_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT
    complex_profile_filter_sys_prompt: str = COMPLEX_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT
    profile_output_instruction: str = PROFILE_OUTPUT_INSTRUCTIONS_DEFAULT
    overridden_caller_func: Optional[str] = None

    # tool arguments metadata
    arg_metadata = {
        "profile_filter_main_prompt": ToolArgMetadata(hidden_from_planner=True),
        "simple_profile_filter_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
        "complex_profile_filter_sys_prompt": ToolArgMetadata(hidden_from_planner=True),
        "profile_output_instruction": ToolArgMetadata(hidden_from_planner=True),
        "overridden_caller_func": ToolArgMetadata(hidden_from_planner=True),
    }


@tool(
    description=FILTER_AND_RANK_STOCKS_BY_PROFILE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
)
async def filter_and_rank_stocks_by_profile(
    args: FilterAndRankStocksByProfileInput, context: PlanRunContext
) -> List[StockID]:
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

    if args.overridden_caller_func:
        caller_func_name = args.overridden_caller_func
    else:
        caller_func_name = filter_and_rank_stocks_by_profile.__name__

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

    # Disabled for ranking as it does not have update logic designed yet
    if args.complete_ranking:
        use_cache = False
    else:
        use_cache = True

    return await run_profile_match(
        args.stocks,
        args.profile,
        args.stock_texts,
        caller_func_name=caller_func_name,
        profile_filter_main_prompt_str=args.profile_filter_main_prompt,
        profile_filter_sys_prompt_str=profile_filter_sys_prompt,
        profile_output_instruction_str=args.profile_output_instruction,
        profile_match_parameters=profile_match_params,
        context=context,
        use_cache=use_cache,
        debug_info=debug_info,
    )


class PerIdeaFilterStocksByProfileMatch(ToolArgs):
    stocks: List[StockID]
    texts: List[StockText]
    ideas: List[Idea]
    profiles: List[TopicProfiles]


@tool(
    description=PER_TOPIC_FILTER_BY_PROFILE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
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
            score_threshold=1,
            overridden_caller_func=per_idea_filter_stocks_by_profile_match.__name__,
        ),
        context=context,
    )  # type: ignore


class PerIdeaFilterAndRankStocksByProfileMatch(ToolArgs):
    stocks: List[StockID]
    stock_texts: List[StockText]
    ideas: List[Idea]
    profiles: List[TopicProfiles]
    complete_ranking: bool
    score_threshold: int = 1
    top_n: Optional[int] = None
    bottom_m: Optional[int] = None

    # prompt arguments (hidden from planner)
    profile_filter_main_prompt: str = PROFILE_FILTER_MAIN_PROMPT_STR_DEFAULT
    simple_profile_filter_sys_prompt: str = SIMPLE_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT
    complex_profile_filter_sys_prompt: str = COMPLEX_PROFILE_FILTER_SYS_PROMPT_STR_DEFAULT
    profile_output_instruction: str = PROFILE_OUTPUT_INSTRUCTIONS_DEFAULT
    overridden_caller_func: Optional[str] = None

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
    category=ToolCategory.LLM_ANALYSIS,
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

    todo_profiles = args.profiles[:]

    removed_stocks: Set[StockID] = set()
    cached_profiles: List[TopicProfiles] = []
    cached_filtered_stocks = {}

    # TODO: doing this both here and inside run_profile_match is redundant, but basically has no
    # effect, and useful to have them available for checking against citations
    split_texts = cast(
        List[StockText], await partition_to_smaller_text_sizes(args.stock_texts, context=context)  # type: ignore
    )

    tasks = []

    if args.overridden_caller_func:
        caller_func_name = args.overridden_caller_func
    else:
        caller_func_name = per_idea_filter_and_rank_stocks_by_profile_match.__name__

    # Disabled for ranking as it does not have update logic designed yet
    if args.complete_ranking is False:
        profile_match_parameters = ProfileMatchParameters()
        try:  # since everything associated with diffing is optional, put in try/except
            prev_run_info = await get_prev_run_info(context, caller_func_name)
            if prev_run_info is not None:
                prev_args = PerIdeaFilterStocksByProfileMatch.model_validate_json(
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
                                    caller_func_name=caller_func_name,
                                    profile_match_parameters=profile_match_parameters,
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
                caller_func_name=caller_func_name,
                profile_match_parameters=profile_match_params,
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
