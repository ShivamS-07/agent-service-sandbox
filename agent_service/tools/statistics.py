import asyncio
import datetime
import json
from typing import List, Optional, Tuple, Union

from agent_service.external.feature_svc_client import get_all_features_metadata
from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable, Table
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.feature_data import (
    StatisticId,
    StatisticsIdentifierLookupInput,
    get_latest_date,
    get_statistic_data,
    statistic_identifier_lookup,
)
from agent_service.tools.tables import (
    JoinTableArgs,
    TransformTableArgs,
    join_tables,
    transform_table,
)
from agent_service.tools.tool_log import tool_log
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import convert_horizon_to_days, get_now_utc
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed

TRADING_TO_CALENDAR_RATIO = 8 / 5

DECOMPOSITION_SYS_PROMPT_STR = "You are a financial analyst who is trying to prepare data to satisfy a client's particular information need. You will be given a reference to a statistic that the user needs, the larger conversational context in which this statistic was mentioned, and a large collection of basic statistics which your database has time series data for. Your goal is to translate this client statistic reference into one or more of the provided basic statistics and an explanation of any mathmatical operations required to translate the selected statistic(s) into the statistic that will satisfy the client. Specifically, you will output a json with three keys. The first key should be called `components` and should be a json list of strings corresponding directly to statistics in the provided list that are useful for calculating the required client-desired statistic. It is okay to have one statistic if that is all you need to fully satisfy the client need. If there appears to be absolutely no way to calculate the desired statistic, components should be an empty list. The second key, `calculation` will be a description of mathematical operations applied to the component statistics that will result in a time series of the desired statistic. This description should be clear and directly refer to any statistics mentioned in the first line. If, based on the chat context, the client need is clearly a time series (i.e the client refer to graphing the statistic over some period), the calculation must transform the input time series into an output time series. If a time series is desired, You must NOT do a calculation which makes sense only for the last date of the time series, turning the time series into a single point! For example, if you are calculating a price change, you must calculate a price change for every day of the time series! To remind you of this, use the term daily with your statistic in your calculation explanation (e.g. daily price delta). If you listed only one statistic in the first key and you are confident that that statistic is in fact exactly what the client wants, you may output an empty string for the `calculation` key, however you must be very confident that the client will be satisfied with this result; if you give them the wrong statistic you will be fired. For the third key, `extra_timespan`, you should consider whether you need to retrieve additional historical time series data for the statistic(s) listed under components to calculate the relevant client-desired statistic. For example, to calculate a 1M moving average, you need 1M of additional data (the first datapoint in the time series requires 1M of data before it). The timespan needed will often (though not always) be mentioned directly in the statistic. If the timespan does, but the statistic requires some kind of window, you can select one that seems reasonable, and include it both your calcuation and extra_timespan. Express the timespan of additional data you need as a string in standard financial format (e.g. 1D, 3M, 1Y), output an empty string if no such additional data is needed (which is very common, for instance you would need no such extra data for standard ratio like Price to Earnings). A few additional guidelines:\n- Generally (unless the client says otherwise), you should interpret `performance` of stocks as referring to their stock price\n- You interpret stock price to mean close price\n- keep percentages in decimal form, do not multiply by 100\n- Any reference to stock price change (gain/loss/delta) should be calculated relative to the first day of the period which is being calculated over\n- If multiple basic statistics seem like they could be used for a specific part of the calculation, you should be biased towards chosing simpler statistics (i.e. those without modifiers)\n- Make sure you are only calculating the provided statistic referenced, you should not include any other calculations mentioned in the chat context\n- Do not output a wrapper around your json (no ```json!)\n- Do not mention the specific stocks from the chat, your output should work for any stock\n- Be concise, unless the calculation is very complex, a single sentence of no more than 40 words is strongly preferred, you do not need a formula."  # noqa: E501

DECOMPOSITION_MAIN_PROMPT_STR = "Identify which of the following list of statistics can be used to calculate the statistic referenced by the user, as understood in the larger chat context, and provide a mathmatical description of how such a derivation would occur. Here is the client mention of the statistic: {statistic_description}\nHere is the larger chat context, delimited by `---`:\n---\n{chat_context}\n---\nAnd here is the long list of statistics you have data for, also delimited by `---`:\n---\n{statistic_list}\n---\nNow output your json consisting of a list of relevant statistics, and an explanation of how to derive the client's statistic from those statistics, and, if applicable, the amount additional time series data that must be requested beyond the timespan asked for by the client:\n"  # noqa: E501

DECOMPOSITION_SYS_PROMPT = Prompt(
    name="DECOMPOSITION_SYS_PROMPT",
    template=DECOMPOSITION_SYS_PROMPT_STR,
)

DECOMPOSITION_MAIN_PROMPT = Prompt(
    name="DECOMPOSITION_MAIN_PROMPT",
    template=DECOMPOSITION_MAIN_PROMPT_STR,
)


class GetStatisticDataForCompaniesInput(ToolArgs):
    statistic_reference: str
    stock_ids: List[StockID]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function returns a time series of data (possibly one point) for a client-provided statistic"
        " for each stock in the list of stock_ids."
        " The function will analyze the statistic reference string, matching it to an appropriate database "
        " key and retrieving the data, or alternatively calculate the data from the data of component statistics."
        " As such, it accepts both simple and complex expressions for financial statistics, including"
        " vague and otherwise underspecified financial variables."
        " Component statistics may include macroeconomic statistics (such as interest rates), however the final "
        " statistic must be calculated on a per-stock basis."
        " If the client wants pure macroeconomic data, use the `get_macro_statistic_data` function."
        " This function only accepts expressions which represent statistics that can be calculated independently for"
        " individual stocks, it cannot be used for calculations across stocks, including averaging across stocks and"
        " filtering. For example, if the client asked `Give me the top 10 stocks in the S&P 500 by the ratio of"
        " average 1M price to EBITDA per share, you can pass `ratio of average 1M price to EBITDA per Share`"
        " as statistic_reference to this tool since it can be calculated independently for each stock in the"
        " S&P 500 (the average refers to averaging across time not stocks!), however you cannot use this function"
        " to rank and filter to the top 10, since that is a cross-stock operation; instead you must use the"
        " `transform_table` function. If the client asked you instead to `graph the average market cap of stocks in"
        " the S&P 500 over the last year`, in that case it is clear that the averaging is across stocks, not time,"
        " and the statistic_reference string should simply be `market cap`, the required averaging must happen in the"
        " transform_table tool, not in this tool."
        " Also, if the client need clearly is a statistic that can be calculated independently for particular stocks"
        " You must use this function with the entire statistic reference. That is, you must call this function once"
        " with the entire expression, not separately for each expression. For instance if the user says they want a"
        " graph of earnings over debt for company X, you should call this function with the reference `earnings over"
        " debt` and NOT call this function independently with `earnings` and `debt`!"
        " You should use this function for general performance indicators (like revenue) which apply to all"
        " stocks, you should use kpi functions only when there are performance indicators involved which are specific"
        " to certain companies and sectors (e.g. iPhone sales, cloud revenue)"
        " If you need the same statistic for the same time period for more than one company, you must call this"
        " function with multiple stock_ids, DO NOT call this function multiple times"
        " with a single stock per time in those circumstances!"
        " Optionally a start_date and end_date may be provided to specify a date range"
        " to get a specific date only then set both start_date and end_date to the same date."
        " if the optional date_range argument is passed in it will override anything set in start_date and end_date "
        " If none of start_date, end_date, date_range are provided then it will assume the request "
        " is for the most recent date for which data exists."
        " If the user does not mention any date or time frame, you should assume they "
        " want the most recent datapoint and call without specifying either start_date or end_date."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
)
async def get_statistic_data_for_companies(
    args: GetStatisticDataForCompaniesInput, context: PlanRunContext
) -> StockTable:

    if context.chat is None:  # for mypy
        raise Exception("No chat context provided")

    logger = get_prefect_logger(__name__)

    stat_ref = args.statistic_reference
    stocks = args.stock_ids
    start_date, end_date = get_start_end_dates(args)

    try:  # try a simple look up first
        logger.info("Trying simple lookup for {stat_ref}")
        stat_id: StatisticId = await statistic_identifier_lookup(  # type: ignore
            args=StatisticsIdentifierLookupInput(statistic_name=stat_ref), context=context
        )
        table: StockTable = await get_statistic_data(context, stat_id, start_date, end_date, stocks)
        return table
    except ValueError as e:
        logger.info(f"Simple lookup failed: {e}, doing LLM decomposition")

    resp = await get_all_features_metadata(context.user_id, filtered=True)
    all_statistic_lookup = {
        feature_metadata.name: feature_metadata
        for feature_metadata in resp.features
        if feature_metadata.importance <= 2
    }
    all_statistics = "\n".join(all_statistic_lookup)
    gpt_context = create_gpt_context(
        GptJobType.AGENT_PLANNER, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)
    main_prompt = DECOMPOSITION_MAIN_PROMPT.format(
        statistic_description=stat_ref,
        chat_context=context.chat.get_gpt_input(),
        statistic_list=all_statistics,
    )
    result = await llm.do_chat_w_sys_prompt(main_prompt, DECOMPOSITION_SYS_PROMPT.format())
    output_json = json.loads(clean_to_json_if_needed(result))
    stat_list = output_json["components"]
    if len(stat_list) == 0:
        raise Exception(
            "No decomposition found for client statistic using supported component statistics"
        )
    calculation = output_json["calculation"]
    added_timespan = output_json["extra_timespan"]
    await tool_log(
        log=f"Analyzed {stat_ref}, component variables: {stat_list}, calculation: {calculation}",
        context=context,
    )
    if added_timespan:
        extra_days = convert_horizon_to_days(added_timespan)
        if (
            "D" in added_timespan and extra_days > 5
        ):  # D refers to trading days, need more calendar days
            extra_days = int(
                extra_days * TRADING_TO_CALENDAR_RATIO
            )  # doesn't hurt to get extra, they'll be droppped
        logger.info(f"Extra days included for calculation: {extra_days} ({added_timespan})")
        comp_start_date = start_date - datetime.timedelta(days=extra_days)
    else:
        comp_start_date = start_date
    force_daily = False
    for (
        stat
    ) in stat_list:  # if any of the component stats are daily, we need to make them all daily
        source = all_statistic_lookup[stat].source
        if "estimate" not in source and "quartery" not in source:
            force_daily = True
    comp_tables = []
    for stat in stat_list:
        stat_id = StatisticId(stat_id=all_statistic_lookup[stat].feature_id, stat_name=stat)
        comp_tables.append(
            await get_statistic_data(
                context=context,
                stock_ids=stocks,
                statistic_id=stat_id,
                start_date=comp_start_date,
                end_date=end_date,
                force_daily=force_daily,
            )
        )

    if len(comp_tables) > 1:
        comp_table: StockTable = await join_tables(  # type: ignore
            JoinTableArgs(input_tables=comp_tables), context  # type: ignore
        )
    else:
        comp_table = comp_tables[0]

    if not calculation:
        logger.warning("No calculation explanation, returning component table")
        return comp_table

    await tool_log("Component statistic data fetched, doing calculation", context=context)

    transformed_table: StockTable = await transform_table(  # type: ignore
        args=TransformTableArgs(input_table=comp_table, transformation_description=calculation),
        context=context,
    )
    if (
        added_timespan
    ):  # if we did a rolling calculation, the extra days are usually there, remove them
        logger.info("Deleting extra days in table due to rolling calculation")
        transformed_table.delete_data_before_start_date(start_date=start_date)
    return transformed_table


class MacroFeatureDataInput(ToolArgs):
    statistic_reference: str
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None
    # in the future we may want to take a currency as well


@tool(
    description=(
        "This function returns the time series of data for a statistic indicated by the"
        " statistic reference string"
        " that is not tied to a specific stock. These are usually macroeconomic indicators like"
        " interest rates, inflation and unemployment rates. If the macroeconomic statistic is "
        " embededed in a more complex, per-stock variable, use `get_statistic_data_for_companies`, "
        " use this function only when the client wants to view macroeconomic variables directly, e.g. "
        " `plot interest rates for the last 5 years`."
        " Optionally a start_date and end_date may be provided to specify a date range"
        " to get a specific date only then set both start_date and end_date to the same date."
        " if the optional date_range argument is passed in it will override anything set in start_date and end_date. "
        " If none of start_date, end_date, date_range are provided then it will assume the request "
        " is for the most recent date for which data exists."
        " If the user does not mention any date or time frame, you should assume they "
        " want the most recent datapoint and call without specifying either start_date or end_date."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
)
async def get_macro_statistic_data(args: MacroFeatureDataInput, context: PlanRunContext) -> Table:
    start_date, end_date = get_start_end_dates(args)
    # Just a sample look up
    # TODO: Add LLM decomposition to this too? Probably much less commmon
    stat_id: StatisticId = await statistic_identifier_lookup(  # type:ignore
        args=StatisticsIdentifierLookupInput(statistic_name=args.statistic_reference),
        context=context,
    )
    return await get_statistic_data(context, stat_id, start_date, end_date)


def get_start_end_dates(
    args: Union[GetStatisticDataForCompaniesInput, MacroFeatureDataInput],
) -> Tuple[datetime.date, datetime.date]:

    if args.date_range:
        args.start_date = args.date_range.start_date
        args.end_date = args.date_range.end_date

    # if no dates given, use latest date
    if args.start_date is None and args.end_date is None:
        latest_date = get_latest_date()
        start_date = latest_date
        end_date = latest_date
    # if only end date given, use end to end
    if args.start_date is None and args.end_date is not None:
        start_date = args.end_date
        end_date = args.end_date
    # if only start date given, use start to latest()
    elif args.start_date is not None and args.end_date is None:
        start_date = args.start_date
        end_date = get_latest_date()
    # if both dates are given use as is
    elif args.start_date is not None and args.end_date is not None:
        start_date = args.start_date
        end_date = args.end_date

    return start_date, end_date


async def main() -> None:
    init_stdout_logging()
    plan_context = PlanRunContext.get_dummy()
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)
    input_text = "Give me a graph of {statistic} for Apple and Microsoft for the last 3 months"
    statistics = [
        "Percent Gain",
        "Price to EBIT",
        "1M Average Daily Volume",
        "60-day momentum",
        "performance",
    ]
    for stat in statistics:
        user_message = Message(
            message=input_text.format(statistic=stat),
            is_user_message=True,
            message_time=get_now_utc(),
        )
        plan_context.chat = ChatContext(messages=[user_message])
        result = await get_statistic_data_for_companies(  # type: ignore
            args=GetStatisticDataForCompaniesInput(
                statistic_reference=stat,
                stock_ids=await StockID.from_gbi_id_list([714, 6963]),
                start_date=start_date,
                end_date=end_date,
            ),
            context=plan_context,
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
