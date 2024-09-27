from typing import List, Optional

from agent_service.io_types.stock import StockID
from agent_service.io_types.text import StockText
from agent_service.planner.errors import EmptyInputError
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.stock_rank_by_text.prompts import (
    RANK_STOCKS_BY_PROFILE_DESCRIPTION,
)
from agent_service.tools.stock_rank_by_text.utils import (
    evaluate_and_summarize_profile_fit_for_stocks,
    get_profile_rubric,
    rank_individual_levels,
    stocks_rubric_score_assignment,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.prefect import get_prefect_logger


class RankStocksByProfileInput(ToolArgs):
    stocks: List[StockID]
    stock_texts: List[StockText]
    profile: str
    top_n: Optional[int] = None
    bottom_m: Optional[int] = None


@tool(
    description=RANK_STOCKS_BY_PROFILE_DESCRIPTION,
    category=ToolCategory.LLM_ANALYSIS,
)
async def rank_stocks_by_profile(
    args: RankStocksByProfileInput, context: PlanRunContext
) -> List[StockID]:
    logger = get_prefect_logger(__name__)

    if len(args.stocks) == 0:
        raise EmptyInputError("Cannot rank empty list of stocks")
    if len(args.stock_texts) == 0:
        raise EmptyInputError("Cannot rank stocks with empty list of texts")

    logger.info("Summarizing Relevant Text For Each Stock...")
    stock_summary_map = await evaluate_and_summarize_profile_fit_for_stocks(
        profile=args.profile,
        texts=args.stock_texts,
        stocks=args.stocks,
        context=context,
    )
    relevant_stock_ids = list(stock_summary_map.keys())

    logger.info("Generating rubric...")
    rubric_dict = await get_profile_rubric(
        args.profile, context.agent_id, dict(list(stock_summary_map.items())[:3])
    )

    logger.info("Applying Rubric...")
    stocks_with_scores = await stocks_rubric_score_assignment(
        relevant_stock_ids, rubric_dict, stock_summary_map, args.profile, context, drop_zeros=False
    )

    logger.info("Applying Inter-level Ranking...")
    fully_ranked_stocks = await rank_individual_levels(
        args.profile, stocks_with_scores, context, top_n=args.top_n
    )
    await tool_log(
        f"Analysed the given stocks for exposure to '{args.profile}' and ranked them in descending order",
        context=context,
    )

    # Use a set for top_n & bottom_m so as to avoid cases where we return duplicate stocks
    # if top_n + bottom_m > len(fully_ranked_stocks)
    truncated_ranked_stocks = set()
    if args.top_n:
        logger.info(f"Determined the top {args.top_n}")
        top_stocks = fully_ranked_stocks[: args.top_n]
        non_zero_top_stocks = [stock for stock in top_stocks if stock.history[-1].score.val != 0]  # type: ignore
        if len(non_zero_top_stocks) == 0:
            await tool_log(
                f"Could not find any relavent stocks from the given set relevant to '{args.profile}'",
                context=context,
            )
        elif len(non_zero_top_stocks) < len(top_stocks):
            await tool_log(
                f"Only able to find {len(non_zero_top_stocks)} top stocks, all other stocks were not relevant",
                context=context,
            )
        else:
            await tool_log(
                f"Determined the top {args.top_n}",
                context=context,
            )
        truncated_ranked_stocks.update(non_zero_top_stocks)
    if args.bottom_m:
        logger.info(f"Determined the bottom {args.bottom_m}")
        await tool_log(
            f"Determined the bottom {args.bottom_m}",
            context=context,
        )
        truncated_ranked_stocks.update(fully_ranked_stocks[args.bottom_m * (-1) :])
    if args.top_n or args.bottom_m:
        truncated_stock_list = sorted(
            list(truncated_ranked_stocks), key=lambda stock: stock.history[-1].score.val, reverse=True  # type: ignore
        )
        return truncated_stock_list
    else:
        return fully_ranked_stocks
