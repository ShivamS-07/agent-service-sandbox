import datetime
import random
from asyncio.log import logger
from collections import defaultdict
from copy import deepcopy
from math import ceil, floor
from typing import Dict, List, Optional, Set, Tuple

from agent_service.GPT.constants import FILTER_CONCURRENCY, GPT4_O, GPT4_O_MINI, SONNET
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import Citation, HistoryEntry, Score
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.text import StockText, Text
from agent_service.tools.LLM_analysis.constants import RUBRIC_DELIMITER
from agent_service.tools.LLM_analysis.utils import extract_citations_from_gpt_output
from agent_service.tools.stock_filter_and_rank.constants import (
    EVALUATE_AND_SUMMARIZE_CONCURRENCY,
    NONRELEVANT_COMPANY_EXPLANATION,
    PAIRWISE_CONCURRENCY,
    RANDOM_SEED,
    SAMPLES_DELIMITER,
    SCORE_MAPPING,
    SCORE_OUTPUT_DELIMITER,
    TIEBREAKER_CONCURRENCY,
)
from agent_service.tools.stock_filter_and_rank.prompts import (
    PROFILE_ADD_DIFF_MAIN_PROMPT,
    PROFILE_ADD_DIFF_SYS_PROMPT,
    PROFILE_EXPOSURE_TEXT_EVALUATER_MAIN_PROMPT,
    PROFILE_EXPOSURE_TEXT_EVALUATER_SYS_PROMPT,
    PROFILE_REMOVE_DIFF_MAIN_PROMPT,
    PROFILE_REMOVE_DIFF_SYS_PROMPT,
    PROFILE_RUBRIC_EXAMPLES_MAIN_INSTRUCTION,
    PROFILE_RUBRIC_EXAMPLES_SYS_INSTRUCTION,
    PROFILE_RUBRIC_GENERATION_MAIN_OBJ,
    PROFILE_RUBRIC_GENERATION_SYS_OBJ,
    RUBRIC_EVALUATION_MAIN_OBJ,
    RUBRIC_EVALUATION_SYS_OBJ,
    TIEBREAKER_MAIN_PROMPT,
    TIEBREAKER_SYS_PROMPT,
    TWO_COMP_PROFILE_COMPARISON_MAIN_PROMPT,
    TWO_COMP_PROFILE_COMPARISON_SYS_PROMPT,
)
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt

random.seed(RANDOM_SEED)


async def profile_filter_stock_match(
    aligned_text_groups: StockAlignedTextGroups,
    str_lookup: Dict[StockID, str],
    profile: str,
    is_using_complex_profile: bool,
    llm: GPT,
    context: PlanRunContext,
    profile_filter_main_prompt: Prompt,
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


async def evaluate_profile_fit_for_stock(
    profile: str,
    stock_id: StockID,
    company_texts: str,
    context: PlanRunContext,
    input_llm: Optional[GPT] = None,
) -> bool:
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )

    if input_llm:
        llm = input_llm
    else:
        llm = GPT(model=GPT4_O_MINI, context=gpt_context)

    tokenizer = GPTTokenizer(model=llm.model)
    fixed_len = tokenizer.get_token_length(
        input=(
            " ".join(
                [
                    PROFILE_EXPOSURE_TEXT_EVALUATER_SYS_PROMPT.template,
                    PROFILE_EXPOSURE_TEXT_EVALUATER_MAIN_PROMPT.template,
                    profile,
                ]
            )
        )
    )
    chopped_company_texts = tokenizer.chop_input_to_allowed_length(
        flexible_input=company_texts, fixed_input_len=fixed_len
    )

    llm_output = await llm.do_chat_w_sys_prompt(
        main_prompt=PROFILE_EXPOSURE_TEXT_EVALUATER_MAIN_PROMPT.format(
            profile=profile,
            company_name=stock_id.company_name,
            company_texts=chopped_company_texts,
        ),
        sys_prompt=PROFILE_EXPOSURE_TEXT_EVALUATER_SYS_PROMPT.format(),
        max_tokens=1,
    )

    try:
        decision = int(llm_output.strip())
    except ValueError:
        decision = 2
        logger.warning(
            f"Failed to proper decifer llm output during stock rank stock evaluator, got {llm_output}"
        )

    if decision == 1:
        return True
    else:
        logger.info(
            f"{stock_id.symbol} ({stock_id.company_name}) did not have any relevant "
            f"information for profile '{profile}'"
        )
        return False


async def evaluate_profile_fit_for_stocks(
    profile: str,
    aligned_text_groups: StockAlignedTextGroups,
    str_lookup: Dict[StockID, str],
    context: PlanRunContext,
    llm: Optional[GPT] = None,
) -> Dict[StockID, Tuple[str, List[Citation]]]:
    tasks = []
    for stock_id, _ in aligned_text_groups.val.items():
        tasks.append(
            evaluate_profile_fit_for_stock(
                profile=profile,
                stock_id=stock_id,
                company_texts=str_lookup[stock_id],
                input_llm=llm,
                context=context,
            )
        )

    res = await gather_with_concurrency(tasks, n=EVALUATE_AND_SUMMARIZE_CONCURRENCY)
    return res


async def compare_stocks(
    stock1: StockID, stock2: StockID, profile: str, context: PlanRunContext
) -> Optional[StockID]:
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(model=GPT4_O_MINI, context=gpt_context)
    llm_output = await llm.do_chat_w_sys_prompt(
        main_prompt=TWO_COMP_PROFILE_COMPARISON_MAIN_PROMPT.format(
            profile=profile,
            company1_name=stock1.company_name,
            company1_summary=stock1.history[-1].explanation,
            company2_name=stock2.company_name,
            company2_summary=stock2.history[-1].explanation,
        ),
        sys_prompt=TWO_COMP_PROFILE_COMPARISON_SYS_PROMPT.format(),
        max_tokens=200,
    )

    gpt_decision = (llm_output.strip())[0]
    if gpt_decision == "1":
        return stock1
    elif gpt_decision == "2":
        return stock2
    else:
        logger.error(
            f"Compare stock failed for {stock1.gbi_id} and {stock2.gbi_id} "
            f"for topic '{profile}', got: {llm_output}"
        )
        return None


async def tie_breaker(profile: str, stocks: List[StockID], context: PlanRunContext) -> List[int]:
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(model=GPT4_O_MINI, context=gpt_context)
    companies_str = [
        f"Company {i + 1}: {stock.company_name}:\n{stock.history[-1].explanation}\n------"
        for i, stock in enumerate(stocks)
    ]
    res = await llm.do_chat_w_sys_prompt(
        main_prompt=TIEBREAKER_MAIN_PROMPT.format(
            profile=profile,
            companies_str=companies_str,
        ),
        sys_prompt=TIEBREAKER_SYS_PROMPT.format(),
    )

    try:
        raw_ranked_order = map(int, (res.strip()).split(","))
        ranked_order = [rank - 1 for rank in raw_ranked_order]
        return ranked_order
    except ValueError:
        logger.error(
            f"Error trying to decode GPT tiebreaker ranking {len(stocks)} companies, got: {res}"
        )
        return []


async def tiebreaker_policy(
    profile: str,
    tied_stocks: List[StockID],
    tied_score: float,
    score_increments: float,
    context: PlanRunContext,
) -> None:
    upper_bound_score = min(ceil(tied_score), tied_score + (score_increments / 2))
    lower_bound_score = max(floor(tied_score), tied_score - score_increments / 2)

    step = (upper_bound_score - lower_bound_score) / (len(tied_stocks) + 1)  # Calculate step size
    new_scores = [lower_bound_score + step * (i + 1) for i in range(len(tied_stocks))]

    # This returns the indicies for the stocks in ranked order
    ranked_order = await tie_breaker(profile, tied_stocks, context)

    for stock_index, new_score in zip(ranked_order, new_scores):
        tied_stocks[stock_index].history[-1].score = Score(val=new_score)


async def apply_tiebreaker_policy(
    profile: str, stocks: List[StockID], score_increments: float, context: PlanRunContext
) -> None:
    # Group stocks by their final rank, assume that the list of stocks is sorted
    # by score in desc order, modify their scores in-place
    final_score_groups: Dict[float, List[StockID]] = {}
    for stock in stocks:
        # All stocks should have a score at this point
        score = stock.history[-1].score.val  # type: ignore
        if score not in final_score_groups:
            final_score_groups[score] = []
        final_score_groups[score].append(stock)

    # For each group of stocks with the same final rank, sort alphabetically by name
    tasks = []
    for tied_score, tied_stocks in final_score_groups.items():
        if len(tied_stocks) > 1:  # Only apply tiebreaker if there are ties
            logger.info(
                f"Tiebreaker applied for stocks with tied score {tied_score}: {len(tied_stocks)}"
            )
            tasks.append(
                tiebreaker_policy(profile, tied_stocks, tied_score, score_increments, context)
            )
    await gather_with_concurrency(tasks, n=TIEBREAKER_CONCURRENCY)


async def run_pairwise_comparison(
    stocks: List[StockID], profile: str, context: PlanRunContext
) -> List[StockID]:
    # Randomly select a max of 10 comparison stocks from the current level and
    # pairwise compare with the rest across each level
    comparison_set = random.sample(stocks, min(10, len(stocks)))
    compared_pairs = set()

    gbi_stock_id_lookup = {stock.gbi_id: deepcopy(stock) for stock in stocks}

    compared_stocks = []
    tasks = []
    for stock in stocks:
        for comparison_stock in comparison_set:
            if stock != comparison_stock:
                if (stock, comparison_stock) not in compared_pairs:
                    tasks.append(compare_stocks(stock, comparison_stock, profile, context))
                    compared_stocks.append((stock, comparison_stock))

                    # Mark this pair as compared
                    compared_pairs.add((stock, comparison_stock))
                    compared_pairs.add((comparison_stock, stock))

    results = await gather_with_concurrency(tasks, PAIRWISE_CONCURRENCY)

    for stocks, winning_stock in zip(compared_stocks, results):
        if winning_stock:
            candidate_stock = stocks[0]
            comparison_stock = stocks[1]

            candidate_gbi = candidate_stock.gbi_id
            comparison_gbi = comparison_stock.gbi_id
            # Adjust score based on comparison result
            non_zero_comparisons = len(comparison_set)
            if candidate_stock in comparison_set:
                non_zero_comparisons -= 1
            if non_zero_comparisons > 0:
                # We also need to update the score of the comparison stock if the candidate stock is also
                # in the list of comparison stocks, otherwise the comparison stock will have far less comparisons
                # than designed
                cand_scr_delta = 0.49 / non_zero_comparisons
                comp_scr_delta = 0.49 / (len(comparison_set) - 1)

                # These stock_ids should always have a score at this point
                if winning_stock == candidate_stock:
                    gbi_stock_id_lookup[candidate_gbi].history[-1].score.val += cand_scr_delta  # type: ignore
                    if candidate_stock in comparison_set:
                        gbi_stock_id_lookup[comparison_gbi].history[-1].score.val -= comp_scr_delta  # type: ignore
                else:
                    gbi_stock_id_lookup[candidate_gbi].history[-1].score.val -= cand_scr_delta  # type: ignore
                    if candidate_stock in comparison_set:
                        gbi_stock_id_lookup[comparison_gbi].history[-1].score.val += comp_scr_delta  # type: ignore
        else:
            # If winning_stock is None, then the pairwise failed between those two stocks, an error log is been made,
            # we will not deem a winner for that comparison
            continue

    inner_level_ranked_stocks = list(gbi_stock_id_lookup.values())
    logger.info("Scores before tiebreaker:")
    logger.info("\n".join([str(stock.history[-1].score.val) for stock in inner_level_ranked_stocks]))  # type: ignore
    # Apply tiebreaker policy for stocks with the same final score
    await apply_tiebreaker_policy(
        profile, inner_level_ranked_stocks, 0.49 / len(comparison_set), context
    )

    logger.info("Scores after tiebreaker:")
    logger.info("\n".join([str(stock.history[-1].score.val) for stock in inner_level_ranked_stocks]))  # type: ignore
    return inner_level_ranked_stocks


async def rank_individual_levels(
    profile: str, stocks: List[StockID], context: PlanRunContext, top_n: Optional[int] = None
) -> List[StockID]:
    stock_score_mapping: Dict[float, List[StockID]] = defaultdict(list)
    fully_ranked_stocks: List[StockID] = []

    # Bucketize the stocks into discrete scores
    for stock in stocks:
        latest_stock_history = stock.history[-1]
        stock_score_mapping[latest_stock_history.score.val].append(stock)  # type: ignore

    for initial_score in sorted(stock_score_mapping.keys(), reverse=True):
        stocks = stock_score_mapping[initial_score]
        # No need to do inner-level ranking for stocks with a 0 score
        if initial_score != 0:
            # Re-initialize each stock with a score at the midpoint
            for stock in stocks:
                # Set the score back to a rating between (0, 5) inclusive
                stock.history[-1].score = Score(val=(initial_score * 5) - 0.5)

            # If theres only one stock in the score then we don't need to do any inner-level ranking
            if len(stocks) < 2:
                fully_ranked_stocks.extend(stocks)
            else:
                ranked_stocks_for_level = await run_pairwise_comparison(stocks, profile, context)
                fully_ranked_stocks.extend(ranked_stocks_for_level)
        else:
            fully_ranked_stocks.extend(stocks)

        # If a user only wants the top N stocks, there's no reason to keep going if we already have at least
        # 10 of the highest ranked stocks
        if top_n:
            if len(fully_ranked_stocks) >= top_n:
                break

    for stock in fully_ranked_stocks:
        # We now need to normalize the score from [0, 5] back to [0, 1]
        stock.history[-1].score.val = stock.history[-1].score.val / 5  # type: ignore

    fully_ranked_stocks = sorted(
        fully_ranked_stocks, key=lambda stock: stock.history[-1].score.val, reverse=True  # type: ignore
    )
    return fully_ranked_stocks


# Rubric Generation & Evaluation Code
async def get_profile_rubric(
    profile: str,
    agent_id: str,
    sample_candidates: Optional[Dict[StockID, Tuple[str, List[Citation]]]] = None,
) -> Dict[int, str]:
    gpt_context = create_gpt_context(GptJobType.AGENT_TOOLS, agent_id, GptJobIdType.AGENT_ID)
    llm = GPT(context=gpt_context, model=SONNET)

    if sample_candidates is not None:
        samples = "\n".join(
            [
                f"Company Name:{stock_id.company_name}\nSummary Below:\n{summary_text[0]}\n{SAMPLES_DELIMITER}"
                for stock_id, summary_text in sample_candidates.items()
            ]
        )

        sys_prompt = PROFILE_RUBRIC_GENERATION_SYS_OBJ.format(
            rubric_delimiter=RUBRIC_DELIMITER,
            additional_instruction=PROFILE_RUBRIC_EXAMPLES_SYS_INSTRUCTION,
        )
        main_prompt = PROFILE_RUBRIC_GENERATION_MAIN_OBJ.format(
            profile=profile,
            additional_instruction=PROFILE_RUBRIC_EXAMPLES_MAIN_INSTRUCTION.format(
                samples=samples, delimiter=SAMPLES_DELIMITER
            ),
        )
    else:
        sys_prompt = PROFILE_RUBRIC_GENERATION_SYS_OBJ.format(
            rubric_delimiter=RUBRIC_DELIMITER,
            additional_instruction="",
        )
        main_prompt = PROFILE_RUBRIC_GENERATION_MAIN_OBJ.format(
            profile=profile, additional_instruction=""
        )

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=main_prompt,
        sys_prompt=sys_prompt,
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


async def stocks_rubric_score_assignment(
    stocks: List[StockID],
    rubric_dict: Dict[int, str],
    stock_text_lookup: Dict[StockID, Tuple[str, List[Citation]]],
    profile: str,
    context: PlanRunContext,
    drop_zeros: bool = True,
) -> List[StockID]:
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=SONNET)
    tasks = []

    rubric_str_list = [f"Level {k}: {v}" for k, v in rubric_dict.items()]

    skipped_stocks = []
    stocks_evaluated = []
    for stock in stocks:
        if stock_text_lookup[stock][0] == NONRELEVANT_COMPANY_EXPLANATION:
            skipped_stocks.append(
                stock.inject_history_entry(
                    HistoryEntry(
                        explanation=NONRELEVANT_COMPANY_EXPLANATION,
                        title=f"Connection to '{profile}'",
                        score=Score(val=0),
                        citations=[],
                        task_id=context.task_id,
                    )
                )
            )
        else:
            tasks.append(
                llm.do_chat_w_sys_prompt(
                    main_prompt=RUBRIC_EVALUATION_MAIN_OBJ.format(
                        company_name=stock.company_name,
                        reason=stock_text_lookup[stock][0],
                    ),
                    sys_prompt=RUBRIC_EVALUATION_SYS_OBJ.format(
                        rubric_str="\n".join(rubric_str_list),
                    ),
                )
            )
            stocks_evaluated.append(stock)
    scores = await gather_with_concurrency(tasks, 20)

    final_scoring_stocks = []
    for stock, score in zip(stocks_evaluated, scores):
        try:
            level_score, _ = score.split(SCORE_OUTPUT_DELIMITER)
        except ValueError:
            logger.warning(f"Failed to extract score for from rubric, got {score}")
            level_score = "0"

        if drop_zeros and (level_score == "0"):
            continue
        else:
            stock_citations = stock_text_lookup[stock][1]
            if stock_citations is None:
                stock_citations = []
            final_scoring_stocks.append(
                stock.inject_history_entry(
                    HistoryEntry(
                        explanation=stock_text_lookup[stock][0],
                        title=f"Connection to '{profile}'",
                        score=Score(val=SCORE_MAPPING[level_score]),
                        citations=stock_citations,
                        task_id=context.task_id,
                    )
                )
            )

    # Add the stocks we skipped due to lack of relevant information from its documents
    final_scoring_stocks.extend(skipped_stocks)
    return final_scoring_stocks
