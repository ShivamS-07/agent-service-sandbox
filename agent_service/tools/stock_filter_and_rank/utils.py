import datetime
import json
import random
from asyncio.log import logger
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil, floor
from typing import Dict, List, Optional, Set, Tuple, Union

from agent_service.GPT.constants import FILTER_CONCURRENCY, GPT4_O, GPT4_O_MINI, SONNET
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import Citation, HistoryEntry, Score
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_aligned_text import StockAlignedTextGroups
from agent_service.io_types.text import (
    StockText,
    Text,
    TextCitation,
    TextGroup,
    TextIDType,
    TopicProfiles,
)
from agent_service.tools.LLM_analysis.constants import (
    CITATION_SNIPPET_BUFFER_LEN,
    RUBRIC_DELIMITER,
)
from agent_service.tools.LLM_analysis.utils import (
    extract_citations_from_gpt_output,
    get_best_snippet_match,
    get_sentences,
    strip_header,
)
from agent_service.tools.stock_filter_and_rank.constants import (
    EVALUATE_AND_SUMMARIZE_CONCURRENCY,
    MAX_RUBRIC_SCORE,
    MAX_UPDATE_CHECK_RETRIES,
    NONRELEVANT_COMPANY_EXPLANATION,
    PAIRWISE_CONCURRENCY,
    RANDOM_SEED,
    SAMPLES_DELIMITER,
    SCORE_MAPPING,
    SCORE_OUTPUT_DELIMITER,
    TIEBREAKER_CONCURRENCY,
    UPDATE_REWRITE_RETRIES,
)
from agent_service.tools.stock_filter_and_rank.prompts import (
    COMPLEX_REWRITE_UPDATE_SYS,
    FILTER_REWRITE_NEGATIVE_STR,
    FILTER_REWRITE_POSITIVE_STR,
    FILTER_UPDATE_CHECK_MAIN,
    FILTER_UPDATE_CHECK_SYS,
    FILTER_UPDATE_REWRITE_MAIN,
    FILTER_UPDATE_TEMPLATE,
    PROFILE_EXPOSURE_TEXT_EVALUATER_MAIN_PROMPT,
    PROFILE_EXPOSURE_TEXT_EVALUATER_SYS_PROMPT,
    PROFILE_RUBRIC_EXAMPLES_MAIN_INSTRUCTION,
    PROFILE_RUBRIC_EXAMPLES_SYS_INSTRUCTION,
    PROFILE_RUBRIC_GENERATION_MAIN_OBJ,
    PROFILE_RUBRIC_GENERATION_SYS_OBJ,
    RUBRIC_EVALUATION_MAIN_OBJ,
    RUBRIC_EVALUATION_SYS_OBJ,
    SIMPLE_REWRITE_UPDATE_SYS,
    TEXT_SNIPPET_RELEVANCY_MAIN_PROMPT,
    TEXT_SNIPPET_RELEVANCY_SYS_PROMPT,
    TIEBREAKER_MAIN_PROMPT,
    TIEBREAKER_SYS_PROMPT,
    TWO_COMP_PROFILE_COMPARISON_MAIN_PROMPT,
    TWO_COMP_PROFILE_COMPARISON_SYS_PROMPT,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed

random.seed(RANDOM_SEED)


@dataclass
class ProfileMatchParameters:
    filter_score_threshold: int = 1
    rank_stocks: bool = False
    top_n: Optional[int] = None
    bottom_m: Optional[int] = None


@dataclass
class CheckOutput:
    stock: StockID
    prev_output: Optional[StockID] = None
    changed: bool = False
    explanation: str = ""
    prev_score: int = 0
    new_score: int = 0
    citations: List[TextCitation] = field(default_factory=list)
    text_relevance_cache: List[Tuple[StockText, bool]] = field(default_factory=list)


async def classify_stock_text_relevancy_for_profile(
    text: str,
    profiles_str: str,
    company_name: str,
    llm: GPT,
) -> bool:

    chopped_text_str = GPTTokenizer(model=llm.model).do_truncation_if_needed(
        truncate_str=text,
        other_prompt_strs=[
            TEXT_SNIPPET_RELEVANCY_MAIN_PROMPT.template,
            TEXT_SNIPPET_RELEVANCY_SYS_PROMPT.template,
            company_name,
            profiles_str,
        ],
    )

    output = await llm.do_chat_w_sys_prompt(
        main_prompt=TEXT_SNIPPET_RELEVANCY_MAIN_PROMPT.format(
            company_name=company_name,
            profiles=profiles_str,
            text_snippet=chopped_text_str,
        ),
        sys_prompt=TEXT_SNIPPET_RELEVANCY_SYS_PROMPT.format(),
        max_tokens=500,
    )

    try:
        decision = int(output.strip()[0])
        if decision == 1:
            return True
        else:
            return False
    except (json.JSONDecodeError, ValueError, IndexError):
        logger = get_prefect_logger(__name__)
        logger.warning(
            f"Failed to get text snippet relevancy output, got '{output}'", exc_info=True
        )
        return False


async def classify_stock_text_relevancies_for_profile(
    texts: List[StockText],
    profiles_str: str,
    context: PlanRunContext,
    text_cache: Optional[Dict[TextIDType, str]] = None,
) -> List[StockText]:
    filtered_texts: List[StockText] = []
    text_strs = await Text.get_all_strs(
        texts, include_header=True, include_timestamps=False, text_cache=text_cache
    )

    llm = GPT(
        model=GPT4_O_MINI,
        context=create_gpt_context(GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID),
    )

    tasks = []
    for i, text in enumerate(texts):
        if isinstance(text, StockText):
            text_str = text_strs[i]
            if not text.stock_id:
                continue
            # If StockText, there must be a stock_id
            company_name = text.stock_id.company_name
            tasks.append(
                classify_stock_text_relevancy_for_profile(
                    text=text_str, profiles_str=profiles_str, company_name=company_name, llm=llm
                )
            )

    results = await gather_with_concurrency(tasks, n=200)
    for i, relevancy_decision in enumerate(results):
        if relevancy_decision:
            filtered_texts.append(texts[i])
    return filtered_texts


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


def convert_score_to_int(score: float) -> int:
    return ceil(score * MAX_RUBRIC_SCORE)


async def check_text_diff(
    stock: StockID,
    new_texts: List[StockText],
    profile_str: str,
    rubric_dict: Dict[int, str],
    prev_output: Optional[StockID],
    llm: GPT,
    context: PlanRunContext,
) -> CheckOutput:
    company_name = stock.company_name
    filtered_new_texts = await classify_stock_text_relevancies_for_profile(
        new_texts, profiles_str=profile_str, context=context  # type: ignore
    )

    filtered_text_set = set(filtered_new_texts)
    text_relevance_cache = [(text, text in filtered_text_set) for text in new_texts]

    if not filtered_new_texts:
        return CheckOutput(stock=stock, changed=False, text_relevance_cache=text_relevance_cache)
    text_group = TextGroup(val=filtered_new_texts)  # type:ignore
    text_str = await Text.get_all_strs(
        text_group, include_header=True, text_group_numbering=True, include_symbols=True
    )
    if not prev_output or not prev_output.history or not prev_output.history[-1].score:
        prev_score = 0.0
        prev_explanation = NONRELEVANT_COMPANY_EXPLANATION
    else:
        prev_score = prev_output.history[-1].score.val * MAX_RUBRIC_SCORE
        prev_explanation = prev_output.history[-1].explanation  # type: ignore

    prev_score_int = ceil(prev_score)
    main_prompt = FILTER_UPDATE_CHECK_MAIN.format(
        company_name=company_name,
        rubric="\n".join([f"Level {k}: {v}" for k, v in rubric_dict.items()]),
        score=prev_score_int,
        explanation=prev_explanation,
        texts=text_str,
    )
    retry = 0
    success = False
    explanation = NONRELEVANT_COMPANY_EXPLANATION
    new_score = 0
    final_citations: List[TextCitation] = []
    while not success:
        try:
            result = await llm.do_chat_w_sys_prompt(main_prompt, FILTER_UPDATE_CHECK_SYS.format())
            score_str, explanation, citation_str = result.strip().split("\n")
            citation_list = json.loads(clean_to_json_if_needed(citation_str))
            new_score = int(score_str)
            if new_score == prev_score_int or explanation == "No change":
                # just return the cache
                return CheckOutput(stock=stock, text_relevance_cache=text_relevance_cache)
            final_citations = []

            for citation_dict in citation_list:
                citation_snippet = None
                citation_snippet_context = None

                cited_text = text_group.convert_citation_num_to_text(int(citation_dict["num"]))
                if cited_text is None:
                    continue
                if "snippet" in citation_dict:
                    citation_snippet = citation_dict["snippet"]
                    source_text_str = text_group.get_str_for_text(cited_text.id)
                    if source_text_str is not None:
                        source_text_str = strip_header(source_text_str)
                        idx = source_text_str.find(citation_snippet)
                        if idx == -1:  # GPT messed up, snippet is not a substring
                            sentences = get_sentences(source_text_str)
                            citation_snippet = await get_best_snippet_match(
                                citation_snippet, sentences, llm
                            )
                            idx = source_text_str.find(citation_snippet)
                        citation_snippet_context = source_text_str[
                            max(0, idx - CITATION_SNIPPET_BUFFER_LEN) : idx
                            + len(citation_snippet)
                            + CITATION_SNIPPET_BUFFER_LEN
                        ]
                else:
                    citation_snippet = None
                    citation_snippet_context = None
                final_citations.append(
                    TextCitation(
                        source_text=cited_text,
                        citation_text_offset=0,  # we set the offset in the differ
                        citation_snippet=citation_snippet,
                        citation_snippet_context=citation_snippet_context,
                    )
                )
            assert len(final_citations) > 0
            success = True

        except Exception as e:
            retry += 1
            if retry == MAX_UPDATE_CHECK_RETRIES:
                logger.warning(f"Failed to load update due to {e}, giving up")
                return CheckOutput(stock=stock, text_relevance_cache=text_relevance_cache)
            else:
                logger.warning(f"Failed to load update due to {e}, retrying")
    return CheckOutput(
        stock=stock,
        prev_output=prev_output,
        changed=True,
        explanation=explanation,
        prev_score=prev_score_int,
        new_score=new_score,
        text_relevance_cache=text_relevance_cache,
        citations=final_citations,
    )


async def do_rewrite(
    check_output: CheckOutput,
    relevant_texts: List[StockText],
    is_complex_profile: bool,
    profile_str: str,
    llm: GPT,
    context: PlanRunContext,
) -> StockID:
    required_text_set = set([citation.source_text for citation in check_output.citations])
    required_texts = []
    optional_texts = []
    for text in relevant_texts:
        if text in required_text_set:
            required_texts.append(text)
        else:
            optional_texts.append(text)

    required_text_group = TextGroup(val=required_texts)  # type: ignore
    required_text_str = await Text.get_all_strs(
        required_text_group, include_header=True, text_group_numbering=True, include_symbols=True
    )
    optional_text_group = TextGroup(val=optional_texts, offset=len(required_texts))  # type: ignore
    optional_text_str = await Text.get_all_strs(
        optional_text_group, include_header=True, text_group_numbering=True, include_symbols=True
    )

    if is_complex_profile:
        sys_prompt = COMPLEX_REWRITE_UPDATE_SYS.format()
    else:
        sys_prompt = SIMPLE_REWRITE_UPDATE_SYS.format()

    if check_output.new_score < check_output.prev_score:
        polarity_str = FILTER_REWRITE_NEGATIVE_STR
    else:
        polarity_str = FILTER_REWRITE_POSITIVE_STR

    main_prompt = FILTER_UPDATE_REWRITE_MAIN.format(
        company_name=check_output.stock.company_name,
        required_texts=required_text_str,
        optional_texts=optional_text_str,
        profile_str=profile_str,
        polarity_str=polarity_str,
        today=(
            context.as_of_date.date().isoformat()
            if context.as_of_date
            else datetime.date.today().isoformat()
        ),
    )

    result = await llm.do_chat_w_sys_prompt(main_prompt, sys_prompt)

    merged_group = TextGroup.join(required_text_group, optional_text_group)

    rationale, citations = await extract_citations_from_gpt_output(result, merged_group, context)

    retries = 0
    while (
        not citations_contain_all_texts(citations, required_texts)
        and retries < UPDATE_REWRITE_RETRIES
    ):
        result = await llm.do_chat_w_sys_prompt(main_prompt, sys_prompt)
        rationale, citations = await extract_citations_from_gpt_output(
            result, merged_group, context
        )
        retries += 1

    return check_output.stock.inject_history_entry(
        HistoryEntry(
            explanation=rationale,
            citations=citations,  # type: ignore
            score=Score(val=check_output.new_score / MAX_RUBRIC_SCORE),
            task_id=context.task_id,
            title=f"Connection to '{profile_str}'",
        )
    )


def citations_contain_all_texts(
    citations: Optional[List[TextCitation]], required_texts: List[StockText]
) -> bool:
    if not citations:
        return False
    citation_source_texts = set([citation.source_text for citation in citations])
    return all([required_text in citation_source_texts for required_text in required_texts])


def reset_relevance_cache_ids(
    text_relevance_cache: List[Tuple[StockText, bool]]
) -> List[Tuple[StockText, bool]]:
    output_cache = []
    for text, is_relevant in text_relevance_cache:
        text.reset_id()
        output_cache.append((text, is_relevant))
    return output_cache


def finalize_updates(
    update_dict: Dict[StockID, CheckOutput],
    final_output_lookup: Dict[StockID, StockID],
    ranking: bool = False,
) -> Dict[StockID, Tuple[str, List[TextCitation]]]:
    # this function is necessary because the initial update prompt works with the integer scores, but
    # the final output must use the final scores
    final_update_dict = {}
    for stock, check_output in update_dict.items():
        if ranking:
            final_score = (
                final_output_lookup[stock].history[-1].score.val * MAX_RUBRIC_SCORE  # type: ignore
                if final_output_lookup[stock].history[-1].score is not None
                else 0.0
            )
            final_score_int = convert_score_to_int(final_score)
            prev_score = (
                check_output.prev_output.history[-1].score.val * MAX_RUBRIC_SCORE
                if (check_output.prev_output and check_output.prev_output.history[-1].score)
                else 0.0
            )
            prev_score_int = convert_score_to_int(final_score)
            pre_phrase = FILTER_UPDATE_TEMPLATE.replace(" Y ", f" {prev_score_int} ").replace(
                " Z ", f" {final_score_int} "
            )
            post_phrase = FILTER_UPDATE_TEMPLATE.replace(" Y ", f" {prev_score} ").replace(
                " Z ", f" {final_score} "
            )
            final_explanation = check_output.explanation.replace(pre_phrase, post_phrase)
        else:
            final_explanation = check_output.explanation
        final_update_dict[stock] = (final_explanation, check_output.citations)
    return final_update_dict


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
    input_upper_bound_score: Optional[float] = None,
    input_lower_bound_score: Optional[float] = None,
) -> None:
    if input_lower_bound_score:
        lower_bound_score = input_lower_bound_score
    else:
        lower_bound_score = max(floor(tied_score), tied_score - score_increments / 2)

    if input_upper_bound_score:
        upper_bound_score = input_upper_bound_score
    else:
        upper_bound_score = min(ceil(tied_score), tied_score + (score_increments / 2))

    step = (upper_bound_score - lower_bound_score) / (len(tied_stocks) + 1)  # Calculate step size
    new_scores = [lower_bound_score + step * (i + 1) for i in range(len(tied_stocks))]
    new_scores = new_scores[::-1]

    # This returns the indicies for the stocks in ranked order
    ranked_order = await tie_breaker(profile, tied_stocks, context)

    for stock_index, new_score in zip(ranked_order, new_scores):
        tied_stocks[stock_index].history[-1].score = Score(val=new_score)


async def tiebreaker_policy_for_fixed_stocks(
    profile: str,
    tied_stocks: List[StockID],
    fixed_stock: StockID,
    tied_score: float,
    score_increments: float,
    context: PlanRunContext,
    input_upper_bound_score: Optional[float] = None,
    input_lower_bound_score: Optional[float] = None,
) -> None:
    if input_lower_bound_score:
        lower_bound_score = input_lower_bound_score
    else:
        lower_bound_score = max(floor(tied_score), tied_score - score_increments / 2)

    if input_upper_bound_score:
        upper_bound_score = input_upper_bound_score
    else:
        upper_bound_score = min(ceil(tied_score), tied_score + (score_increments / 2))

    # Call tie_breaker to get the ranked order
    ranked_order = await tie_breaker(profile, tied_stocks, context)

    # Determine where the fixed stock is in the ranked order
    fixed_index = ranked_order.index(tied_stocks.index(fixed_stock)) if fixed_stock else -1

    # Create scores for the stocks that will be above and below the fixed_stock
    above_step = (
        (upper_bound_score - tied_score) / (len(ranked_order) - fixed_index - 1)
        if fixed_index < len(ranked_order) - 1
        else 0
    )
    below_step = (tied_score - lower_bound_score) / fixed_index if fixed_index > 0 else 0

    # Assign scores based on ranking, splitting around the fixed stock
    for idx, stock_index in enumerate(ranked_order):
        if stock_index == fixed_index:
            # Fixed stock keeps the tied_score
            continue
        elif idx < fixed_index:
            # Assign scores below the fixed stock
            new_score = tied_score - (below_step * (fixed_index - idx))
            tied_stocks[stock_index].history[-1].score = Score(val=new_score)
        else:
            # Assign scores above the fixed stock
            new_score = tied_score + (above_step * (idx - fixed_index))
            tied_stocks[stock_index].history[-1].score = Score(val=new_score)


async def apply_tiebreaker_policy(
    profile: str,
    stocks: List[StockID],
    score_increments: float,
    fixed_gbi_ids: Set[int],
    context: PlanRunContext,
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

    final_score_groups = dict(sorted(final_score_groups.items(), reverse=True))
    sorted_scores = list(final_score_groups.keys())

    # For each group of stocks with the same final rank, sort alphabetically by name
    tasks = []
    for i, (tied_score, tied_stocks) in enumerate(final_score_groups.items()):
        if len(tied_stocks) > 1:  # Only apply tiebreaker if there are ties
            logger.info(
                f"Tiebreaker applied for stocks with tied score {tied_score}: {len(tied_stocks)}"
            )

            upper_bound_score = None
            lower_bound_score = None

            # If a fixed set of gbi_ids are passed in, this is being run as part of
            # an update, this means score_increments may not be reliable due to potentially multiple
            # past runs on different sized lists of stocks, safer to just use the neighboring scores
            # as an upper and lower bound
            if len(fixed_gbi_ids):
                if i + 1 < len(sorted_scores):
                    upper_bound_score = sorted_scores[i + 1]
                if i - 1 >= 0:
                    lower_bound_score = sorted_scores[i - 1]

            fixed_stock: Optional[StockID] = None
            for stock in tied_stocks:
                if stock.gbi_id in fixed_gbi_ids:
                    fixed_stock = stock
                    break

            if fixed_stock:
                tasks.append(
                    tiebreaker_policy_for_fixed_stocks(
                        profile,
                        tied_stocks,
                        fixed_stock,
                        tied_score,
                        score_increments,
                        context,
                        upper_bound_score,
                        lower_bound_score,
                    )
                )
            else:
                tasks.append(
                    tiebreaker_policy(
                        profile,
                        tied_stocks,
                        tied_score,
                        score_increments,
                        context,
                        upper_bound_score,
                        lower_bound_score,
                    )
                )
    await gather_with_concurrency(tasks, n=TIEBREAKER_CONCURRENCY)


async def run_pairwise_comparison(
    stocks: List[StockID], fixed_stocks: List[StockID], profile: str, context: PlanRunContext
) -> List[StockID]:
    # Set of stocks we want to fix the input score for
    fixed_gbi_ids = set([stock.gbi_id for stock in fixed_stocks])

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

    # Initialize the ranked stocks with the fixed stocks and add in the new unfixed stocks that have been ranked
    inner_level_ranked_stocks: List[StockID] = fixed_stocks
    for stock in list(gbi_stock_id_lookup.values()):
        if stock.gbi_id not in fixed_gbi_ids:
            inner_level_ranked_stocks.append(stock)

    inner_level_ranked_stocks = sorted(
        inner_level_ranked_stocks, key=lambda stock: stock.history[-1].score.val, reverse=True  # type: ignore
    )

    logger.info("Scores before tiebreaker:")
    logger.info("\n".join([str(stock.history[-1].score.val) for stock in inner_level_ranked_stocks]))  # type: ignore
    # Apply tiebreaker policy for stocks with the same final score
    await apply_tiebreaker_policy(
        profile, inner_level_ranked_stocks, 0.49 / len(comparison_set), fixed_gbi_ids, context
    )

    logger.info("Scores after tiebreaker:")
    logger.info("\n".join([str(stock.history[-1].score.val) for stock in inner_level_ranked_stocks]))  # type: ignore
    return inner_level_ranked_stocks


def dedup_stocks(
    stocks: List[StockID],
) -> List[StockID]:
    # Sort stocks by gbi value to ensure determinism
    sorted_stocks_by_gbi_value = sorted(stocks, key=lambda StockID: StockID.gbi_id)

    # dedup stocks and drop scores lower than the threshold
    company_names = set()
    dedup_res = []
    for stock in sorted_stocks_by_gbi_value:
        if stock.company_name not in company_names:
            company_names.add(stock.company_name)
            dedup_res.append(stock)
    return dedup_res


def apply_score_threshold(
    stocks: List[StockID],
    score_threshold: int,
) -> List[StockID]:
    filtered_result = []
    for stock in stocks:
        try:
            # Need to multiple by the max rubric score to convert from the 0-1 score to the 0-5 level system
            adjusted_score = (stock.history[-1].score.val) * MAX_RUBRIC_SCORE  # type: ignore
            if adjusted_score > score_threshold:
                filtered_result.append(stock)
        except ValueError:
            logger.warning(
                f"{stock.company_name} ({stock.gbi_id}) had no score during profile match, {stock.history[-1]}"
            )
    return filtered_result


async def rank_individual_levels(
    profile: str,
    stocks: List[StockID],
    context: PlanRunContext,
    fixed_stocks: List[StockID] = [],
) -> List[StockID]:
    logger.info("Applying inter-level ranking to individually rank all stocks...")

    stock_score_mapping: Dict[float, List[StockID]] = defaultdict(list)
    fixed_stock_score_mapping: Dict[float, List[StockID]] = defaultdict(list)
    fully_ranked_stocks: List[StockID] = []

    fixed_gbi_ids = set([stock.gbi_id for stock in fixed_stocks])

    # Bucketize the stocks into discrete scores
    for stock in stocks + fixed_stocks:
        latest_stock_history = stock.history[-1]
        # Floor since these scores may have been already assigned in the case of updates
        bucketized_score = ceil(latest_stock_history.score.val * 5) / 5  # type: ignore
        if stock.gbi_id in fixed_gbi_ids:
            fixed_stock_score_mapping[bucketized_score].append(stock)  # type: ignore
        stock_score_mapping[bucketized_score].append(stock)  # type: ignore

    tasks = []
    for initial_score in sorted(stock_score_mapping.keys(), reverse=True):
        # Skip ranking for stocks with 0 scores
        if initial_score == 0:
            fully_ranked_stocks.extend(stocks)
            continue

        stocks = stock_score_mapping[initial_score]
        fixed_stocks_for_score = deepcopy(fixed_stock_score_mapping.get(initial_score, []))

        # If theres only one stock in the score then we don't need to do any inner-level ranking
        if len(stocks) < 2:
            fully_ranked_stocks.extend(stocks)
        else:
            # Re-initialize each stock with a score at the midpoint
            for stock in stocks:
                # Set the score back to a rating between (0, 5) inclusive and shift it to the midpoint
                # to allow it to move up or down during ranking
                stock.history[-1].score = Score(val=ceil(stock.history[-1].score.val * 5) - 0.5)  # type: ignore
            for stock in fixed_stocks_for_score:
                # Set the fixed scores back to a rating between (0, 5) inclusive
                stock.history[-1].score = Score(val=(stock.history[-1].score.val * 5))  # type: ignore
            tasks.append(run_pairwise_comparison(stocks, fixed_stocks_for_score, profile, context))

    # Max tasks will ever be 5
    result = await gather_with_concurrency(tasks, n=5)
    for ranked_stocks_for_level in result:
        fully_ranked_stocks.extend(ranked_stocks_for_level)

    fully_ranked_stocks = sorted(
        fully_ranked_stocks, key=lambda stock: stock.history[-1].score.val, reverse=True  # type: ignore
    )

    for stock in fully_ranked_stocks:
        # We now need to normalize the score from [0, 5] back to [0, 1]
        stock.history[-1].score.val = stock.history[-1].score.val / 5  # type: ignore

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
            level_score = "1"
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


async def apply_ranking_parameters(
    ranked_stocks: List[StockID],
    profile: Union[str, TopicProfiles],
    context: PlanRunContext,
    top_n: Optional[int] = None,
    bottom_m: Optional[int] = None,
) -> List[StockID]:
    # Use a set for top_n & bottom_m so as to avoid cases where we return duplicate stocks
    # if top_n + bottom_m > len(fully_ranked_stocks)
    truncated_ranked_stocks = set()
    if top_n:
        logger.info(f"Determined the top {top_n}")
        top_stocks = ranked_stocks[:top_n]
        non_zero_top_stocks = [stock for stock in top_stocks if stock.history[-1].score.val != 0]  # type: ignore

        if len(non_zero_top_stocks) == 0:
            profile_topic = profile if isinstance(profile, str) else profile.topic
            await tool_log(
                "Could not find any relevant stocks from the given set relevant "
                f"to '{profile_topic}'",
                context=context,
            )
        elif (len(non_zero_top_stocks) < len(top_stocks)) or (len(non_zero_top_stocks) < top_n):
            await tool_log(
                f"Only able to find {len(non_zero_top_stocks)} top stocks, "
                "all other stocks were not relevant",
                context=context,
            )
        else:
            await tool_log(
                f"Determined the top {top_n}",
                context=context,
            )
        truncated_ranked_stocks.update(non_zero_top_stocks)
    if bottom_m:
        logger.info(f"Determined the bottom {bottom_m}")
        await tool_log(
            f"Determined the bottom {bottom_m}",
            context=context,
        )
        truncated_ranked_stocks.update(ranked_stocks[bottom_m * (-1) :])
    if top_n or bottom_m:
        truncated_stock_list = sorted(
            list(truncated_ranked_stocks),
            key=lambda stock: stock.history[-1].score.val,  # type: ignore
            reverse=True,
        )
        return truncated_stock_list
    else:
        non_zero_ranked_stocks = [
            stock for stock in ranked_stocks if stock.history[-1].score.val != 0  # type: ignore
        ]
        return non_zero_ranked_stocks
