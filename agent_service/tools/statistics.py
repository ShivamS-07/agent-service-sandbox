import asyncio
import datetime
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pytz

from agent_service.external.feature_svc_client import (
    get_all_variables_metadata,
    get_intraday_prices,
)
from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry, TableColumnType
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    StockTable,
    StockTableColumn,
    Table,
    TableColumn,
    TableColumnMetadata,
)
from agent_service.planner.errors import EmptyOutputError
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
from agent_service.utils.date_utils import (
    convert_horizon_to_days,
    get_next_quarter,
    get_now_utc,
    get_prev_quarter,
)
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed

TRADING_TO_CALENDAR_RATIO = 8 / 5

DEFAULT_TIME_DELTA = datetime.timedelta(days=365)
ONE_DAY = datetime.timedelta(days=1)
QUARTER = datetime.timedelta(days=92)

EASTERN_TZ = pytz.timezone("US/Eastern")

REMOVE_STATS = set(["Analyst Expectations", "Earnings Growth"])

DECOMPOSITION_SYS_PROMPT_STR = "You are a financial analyst who is trying to prepare data to satisfy a client's particular information need. You will be given a reference to a statistic that the user needs, the larger conversational context in which this statistic was mentioned, and a large collection of basic statistics which your database has time series data for. Your goal is to translate this client statistic reference into one or more of the provided basic statistics and, if necessary, an explanation of any mathematical operations required to translate the selected statistic(s) into the statistic that will satisfy the client. Specifically, you will output a json with three keys. The first key should be called `components` and should be a json list of strings corresponding directly to statistics in the provided list that are either are, or are useful for calculating the required client-desired statistic. If the statistic that is being asked for is directly available in your list of statistics, and it clearly all you need to fully satisfy the client need, do not make your derivation needlessly complicated, you will be fired if you calculate a statistic you already have. Note that the client will not always use exactly the same wording, for instance they will talk about 'divided yield' but the corresponding listed statistic is Common Dividend Yield. Still, it is preferable to pick basic statistics that exactly match statistics mentioned in the input, when possible (e.g. if the client asks for ratio like price to earnings, you would choose price to earnings directly from the list rather than stock price and earnings separately). You should always search the full list carefully for an exact match before settling on a near match. Never, ever make matches to statistics that involve parentheses, e.g. 'Net Income (Excl. Excep & GW)', unless the contents of the parentheses are specifically mentioned by your client; otherwise statistics with parentheses should be your last choice, you must look very carefully for a version without parentheses! if you pick a near match when an exact match exists, you will be fired! You should never, ever need multiple statistics like X High, X Low, and X Median unless they are explicitly asked for (they are NOT useful for delta calculations, which should involve calculation over a time series of a single statistic); just pick one! Note that if the client asks for 'estimates' or 'expectations' for a standard financial statistic, you should look for a 'consensus mean' for that statistic in the list, e.g. for `analyst expectations of EBIT` you would select `EBIT Consensus Mean`. If there is no other context, analyst expectations should be interpreted as 'Target Price Consensus Mean. However, if the client does NOT ask for an estimate or expectation, you must NEVER choose a statistic with Consensus in the name. If there appears to be absolutely no way to get the desired statistic from the provided basic statistics, components should be an empty list. The second key, `calculation` will be an empty string if you have already found the exact statistic you need, otherwise it will be a description of any mathematical operations that must be applied to the component statistics to get the desired statistic (either an individual value for each stock, or a time series, depending on is_timeseries). This description should be clear and specific and directly refer to any statistics mentioned in the first line. If the client need is a time series, the calculation must transform the input time series into an output time series, you must NOT do a calculation which makes sense only for the last date of the time series, turning the time series into a single point! For example, if you are calculating a price change graph, you must calculate a price change for every day of the time series! To remind you of this, you MUST explicitly say in your calculation that you are generating a time series between the two provided dates, you must use the terms `generating` and `time series` and state the two dates. Again, for some needs, including ranking, filtering tasks, a timeseries is NOT required. In these cases you should NOT use the terms daily, weekly, or monthly, and you should only output a single datapoint per stock! Mention in your calculation description that you are outputing a single datapoint for the provided date. As much as possible, you should pass the client's original wording to the calculation function; if the input uses `over the last week` or `month`, you MUST output `over the last week` or `month` in your calculation instructions. It is critical, in particular, that your reference to the output variable in the calculation uses exactly the same wording as the client did (e.g., if the client said `performance`, your calculation description must indicate that you are calculating `performance`, and not only refer to your interpretation of performance). If you listed only one statistic in the `components` key and you are fairly confident that that statistic is in fact exactly what the client wants, you must output an empty string for the `calculation` key, however you should be very confident that the client will be satisfied with this result. Remember you must provide ONLY the statistic that is asked for; sometimes the chat content will involve a larger statistic calculation, but you must ignore that larger calculation if it is not directly relevant for producing the statistic asked for.  For the third key, `extra_timespan`, you should consider whether you need to retrieve additional historical time series data for the statistic(s) listed under components to calculate the relevant client-desired statistic. For example, to calculate a 1M moving average, you need 1M of additional data (the first datapoint in the time series requires 1M of data before it), and to calculate a delta for one year (e.g performance gain over the last year) you will need an extra year of data. If the client is asking for a single data point output but the required calculation requires more dates than that one point, you must include an extra_timespan that covers all relevant dates! The timespan needed will often (though not always) be mentioned directly in the statistic. One case where you MUST add an extra_timespan when the client needs performance number(s) for a specific month/year/multiyear span; in such a case, you MUST include extra timespan corresponding to the span of time indicated. For example, if the user is looking for performance in 2021, you MUST include 1Y of extra timespan, otherwise you will NOT have the data to do the calculation. If the statistic requires some kind of window but the client has not said so explcitly, you can select one that seems reasonable, and include it both your calcuation and extra_timespan. Express the timespan of additional data you need as a string in standard financial format (e.g. 1D, 3M, 1Y), output an empty string if no such additional data is needed (which is very common, for instance you would need no such extra data for standard ratio like Price to Earnings). A few additional guidelines:\n- You should interpret stock price to mean close price\n- Any reference to stock price change (gain/loss/delta) or returns should default to a percentage price change and must be calculated as a cumulative percent change over the period, calculated relative to the first day of the period: you must be explicit in your calculation description that you are doing a cumulative percentage price change calculation, and what the first day is (if you are creating a time series, you must state the reference date independently of the mention of the output time series range, even though the first date is almost always the same!)\n-- However, if the client specifically mentions daily performance, return, or price change, that means the percentage change of close price for each day relative to the previous day, do NOT use a cumulative calculation in that case. You must never use the Daily Returns statistic nor Open Price statistic in calculations of returns (or performance), you must calculate all daily returns using the difference between Close Price on consecutive days. This is very important!\n- You must keep all percentages in decimal form; never, ever multiply them by 100!\n-When there is ambiguity, you should always interpret weeks, months, and years as spans of time relative to the present rather than a specific date/year/month. If the user says `for/over the last month/week`, that means a time ranging consisting of the thirty/seven (calendar) days prior to today and NOT since the first day of the last month/week. 'Over the last week` means the last seven days, not  You must never jump to the conclusion that the client is talking about the first/last day of a week/month/year unless that is exactly the wording used!\n- If multiple basic statistics seem like they could be used for a specific part of the calculation, you should be biased towards chosing simpler statistics (i.e. those without modifiers)\n- LTM in this context means last twelve months, whereas our balance sheet data is generally quarterly: if someone asks for an LTM/annual balance sheet statistic, you should get 12 month of data for extra timespan and sum the results, e.g. Earnings LTM is a sum of twelve months of Earnings\n- If someone asks for total return, we also want a percentage change, but when including the dividend amount you must make sure to normalize the amounts relevant to the stock price at the beginning of the period and, if the user wants a time series, use a cumulative sum of the dividend amounts (or just a regular sum if not a time series). You must explicitly mention both these steps in your calculation description. Do not do this calcuation if the client just asks for 'returns', use it only if `total returns` (or dividend-adjusted returns. etc.) is explicitly mentioned\n- If the client mentions wanting growth of some statistic over some time period, you must always calculate a percentage growth yourself from the raw statistic, do not use any statisics with 'growth' in the name\n- If the client mentions wanting a single datapoint for a statistic that requires some calculation over a time range, but does not make that time range explicit in the chat context, you should default to one year, and mention it in your calculation description and add 1Y of extra data using extra_timespan\n- Make sure you are only calculating the provided statistic referenced, you should not include any other calculations mentioned in the chat context, in particular do not mention anything related to any filtering or ranking in your calculation description, another analyst will handle that; your task is limited to deriving the statistic mentioned by the client\n- Do not output a wrapper around your json (no ```json!)\n- Do not mention the specific stocks from the chat, your output should work for any stock\n- Be concise, unless the calculation is very complex, a single sentence of no more than 40 words is strongly preferred, you do not need a formula."  # noqa: E501

TWO_QUARTERS = datetime.timedelta(days=183)

# These EPS stats don't work for all companies, so need to be fixed
ACTUAL_EPS = "Net EPS - Basic"
EXPECTED_EPS = "EPS Normalized Consensus Median"

ACTUAL_REVENUE = "Revenues"
EXPECTED_REVENUE = "Revenue Consensus Median"

DECOMPOSITION_MAIN_PROMPT_STR = "Identify which of the following list of statistics is, or can be used to derive the statistic referenced by the user, as understood in the larger chat context, and provide a mathematical description of how such a derivation would occur, as needed. {time_series}\n Here is the statistic you must return: {statistic_description}\nHere is the larger chat context, delimited by `---`:\n---\n{chat_context}\n---\nAnd here is the long list of statistics you have data for, also delimited by `---`:\n---\n{statistic_list}\n---\nNow output your json consisting of a list of relevant statistics, and an explanation of how to derive the client's statistic from those statistics, and, if applicable, the amount additional time series data that must be requested beyond the timespan asked for by the client:\n"  # noqa: E501

DECOMPOSITION_SYS_PROMPT = Prompt(
    name="DECOMPOSITION_SYS_PROMPT",
    template=DECOMPOSITION_SYS_PROMPT_STR,
)

DECOMPOSITION_MAIN_PROMPT = Prompt(
    name="DECOMPOSITION_MAIN_PROMPT",
    template=DECOMPOSITION_MAIN_PROMPT_STR,
)

TIME_SERIES_TEMPLATE = (
    "Note that you are generating a time series between {start_date} and {end_date} for each stock."
)

SINGLE_DATE_TEMPLATE = (
    "Note that you are generating a single point of data for {date} for each stock."
)


async def get_statistic_lookup(context: PlanRunContext) -> Dict[str, Any]:

    resp = await get_all_variables_metadata(context.user_id)
    all_statistic_lookup = {
        feature_metadata.name: feature_metadata
        for feature_metadata in resp.features
        if feature_metadata.importance <= 2 and feature_metadata.name not in REMOVE_STATS
    }
    return all_statistic_lookup


class GetStatisticDataForCompaniesInput(ToolArgs):
    statistic_reference: str
    stock_ids: List[StockID]
    date_range: Optional[DateRange] = None
    is_time_series: bool = False
    target_currency: Optional[str] = None


@tool(
    description=(
        "This function returns a time series of data (possibly one point) for a client-provided statistic"
        " for each stock in the list of stock_ids."
        " The function will analyze the statistic reference string, matching it to an appropriate database"
        " key and retrieving the data, or alternatively calculate the data from the data of component statistics."
        " As such, it accepts both simple and complex expressions for financial statistics, including"
        " vague and otherwise underspecified financial variables such as stock `performance`. You MUST let this"
        " function interpret the meaning of statistics, the statistic_reference passed to this function should"
        " usually be copied verbatim from the client input (except for the one case mentioned below), avoid"
        " paraphrasing or copying from a sample plan! If the statistic reference contains a time span, that must"
        " also be preserved, do not change the wording of the time reference!"
        " There is one key exception to always copying word-for-word: If the client mentions a 'negative' statistic"
        " (e.g. performance drop), you must use a neutral formulation of the underlying quantity (e.g. performance"
        " change). Avoid situations where the client might get confused about what positive or negative means for"
        " the output statistic."
        " I repeat: never, ever use negative words like 'drop', 'decline', 'fall', etc. in the statistic_reference"
        " passed to this tool! I'm serious, listen to me or you will be fired."
        " Component statistics may include macroeconomic statistics (such as interest rates), however the final"
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
        " However, in nearly all cases where the client mentions a complex statistic involving more than one basic"
        " statistic, you should call this tool with with the entire expression, not separately for each statistic."
        " Suppose we have any basic statistics like earnings and prices, for any of the following mathmatical"
        " constructions and many more you should pass in the entire expression as the statistic reference and not just"
        " price or earnings individually: Ratio of earnings to price; earnings over price, earnings times price,"
        " the difference of earnings and price; the sum of earnings and price, etc."
        " The general rule is: when two basic statistics are directly combined by a mathematical operator into"
        " some complex statistic, you must pass the full expression as the statistic reference, not the parts!"
        " However, that is only true when there is one combined statistic. If the user lists statistics"
        " e.g. `Show me market cap, P/E, and current stock price for major tech companies`, you must call this"
        " function separately for each statistic, since the individual statistics are not being combined"
        " into a single complex statistic, but are being displayed separately. This tool will only return one "
        " particular statistic; if you need multiple statistics, you must call this tool multiple times."
        " Again, if the client asks for a long list of individual statistics (e.g. `Revenue, EBITDA, Net Income, "
        " Total Assets, ...'), you absolutely must call this tool once for each statistic, and join the results "
        " with the join_table tool, this is extremely important!!!"
        " You cannot, for instance, get both open AND close prices in a single call to this tool. However, again, if "
        " both statistics are used to derive a single statistic, i.e. the spread between open and close prices, then "
        " you CAN use this tool, since the output is the spread (a single number), and not the individual "
        " open/close prices."
        " A very tricky case involves change/deltas. If a user asks for price change over the past week, the "
        " the statistic_reference should be `price change over the past week` and the date range input"
        " input to this function should be a single day (today), since you are only asking for one datapoint"
        " namely today's change relative to a week ago. This function will retrieve the other data needed to"
        " calculate the change for the day, the input date range should correspond to the dates you want data"
        " for, not the dates needed to calculate the data!"
        " Note that if the client asks for a statistic for a time in the past, you must both provide a date_range"
        " corresponding to that past time AND you must mention the relevant time period in the statistic description."
        " For example, if the client asks for 2021 performance of a stock, you would include a date_range for 2021 and"
        " the statistic_reference `Performance for 2021`. If the client mentions an absolute date range at all, you"
        " must assume it is in the past, and create a date range for it, even when is_time_series=False and you are"
        " mentioning it in the statistic_reference. This is very important! if a date_range is not passed in to this"
        " tool in these situations, your calculation will fail and you will be fired!"
        " Statistics that involve averaging or other amalgamation of base statistics over time should also be passed "
        " into this function, e.g. `average P/E over the last 3 years` is also a perfectly valid statistic_reference."
        " In such cases there is no time series and the date_range should be today (default) "
        " You should use this function for general performance indicators (like revenue) which apply to all"
        " stocks, you should use kpi functions only when there are performance indicators involved which are specific"
        " to certain companies and sectors (e.g. iPhone sales, cloud revenue, China revenue)"
        " This tool should not be used to retrieve earnings or KPI data. Use earnings functions for anything"
        " related to earnings. Use kpi functions for anything related to production, units sold, etc. this tool"
        " cannot provide any information that involves counts of products."
        " This function only works with actuals. If the client asks for estimates or projected results"
        " must use kpi tools, not this tool, even if the statistic is not specific to companies "
        " If you need the same statistic for the same time period for more than one company, you must call this"
        " function with multiple stock_ids, DO NOT call this function multiple times"
        " with a single stock per time in those circumstances!"
        " The is_time_series argument should be selected based on whether a time series is the desired output, or"
        " only a single number is needed for each stock. If the data is going to be used in a line graph,"
        " you must pass is_time_series=True to this function. Even if a line graph is not explicitly asked"
        " for, any mention of wanting to see some statistic where a time period is mention is a strong indication"
        " that a time_series  is required, and is_time_series should be True."
        " is_time_series should NEVER be set to True when the data will be used for filtering or rankings of stocks,"
        " in such cases you should only be outputting one number per stock to filter on."
        " If no date_range is provided then this tool will select the range based on whether the output is"
        " a time series or not."
        " If it is a time series, the range will default to a year ending today, and if instead a single date is"
        " required it will assume the request is for the most recent date for which data exists."
        " You must not get a date range if the client has not specified one in their request, just use the"
        " default!"
        " This tool does NOT generally handle news sentiment scores or quant ratings, use the get_stock_recommendations"
        " tool for that. Also words like 'score(s)' and 'ratings(s)' alone must NEVER be passed to this function"
        " as the statistic reference. If a client is using words like that without any other context, you must assume"
        " they want the output of the recommendations tool. However, if the user asks specifically for `analyst"
        " expectations` (using those words), use this tool, not the recommendations tool!"
        " Optionally a target_currency can be provided as a 3 character ISO 4217 code such as 'EUR', 'JPY', 'CAN'"
        " to convert any currency-valued data to the target_currency."
        " If target_currency is None (the default) then the currency of the outputs will be USD. "
        " Use target_currency if the client clearly wants output in a particular foreign currency or if they ask for"
        " a filtering based on a foreign currency, for example if the client asks for `stocks in TSX with"
        " market cap greater than 100B CAD'. When the client mentions a currency amount, you should assume it is USD"
        " unless another currency is specifically discussed."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
    retries=1,
)
async def get_statistic_data_for_companies(
    args: GetStatisticDataForCompaniesInput, context: PlanRunContext
) -> StockTable:

    if context.chat is None:  # for mypy
        raise Exception("No chat context provided")

    logger = get_prefect_logger(__name__)

    stat_ref = args.statistic_reference
    stocks = args.stock_ids

    latest_date = get_latest_date()

    is_timeseries = args.is_time_series

    if args.date_range:
        start_date, end_date = args.date_range.start_date, args.date_range.end_date
        if end_date > latest_date:
            # no end date past the latest date, preserve start/end diff
            end_date = latest_date
            start_date = end_date - (args.date_range.end_date - args.date_range.start_date)
        if start_date == end_date and is_timeseries:
            # we are doing a time series, so we'd better have a range
            start_date = end_date - DEFAULT_TIME_DELTA
        elif start_date != end_date and not is_timeseries:
            # we are not doing a time series, so only want one day
            latest_date = get_latest_date()
            if end_date > latest_date:
                end_date = latest_date
            start_date = end_date
    else:
        if is_timeseries:
            end_date = latest_date
            start_date = latest_date - DEFAULT_TIME_DELTA
        else:
            end_date = latest_date
            start_date = latest_date

    all_statistic_lookup = await get_statistic_lookup(context)
    # sorting by length encourages it to pick simpler stats (seems to start looking at top)
    all_statistics = "\n".join(sorted(all_statistic_lookup, key=lambda x: len(x)))
    if is_timeseries:
        time_series_str = TIME_SERIES_TEMPLATE.format(start_date=start_date, end_date=end_date)
    else:
        time_series_str = SINGLE_DATE_TEMPLATE.format(date=start_date)
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)
    main_prompt = DECOMPOSITION_MAIN_PROMPT.format(
        statistic_description=stat_ref,
        chat_context=context.chat.get_gpt_input(),
        statistic_list=all_statistics,
        time_series=time_series_str,
    )
    result = await llm.do_chat_w_sys_prompt(main_prompt, DECOMPOSITION_SYS_PROMPT.format())
    output_json = json.loads(clean_to_json_if_needed(result))
    stat_list = output_json["components"]
    if len(stat_list) == 0:
        main_prompt = DECOMPOSITION_MAIN_PROMPT.format(
            statistic_description=stat_ref,
            chat_context=stat_ref,
            statistic_list=all_statistics,
            time_series=time_series_str,
        )
        result = await llm.do_chat_w_sys_prompt(main_prompt, DECOMPOSITION_SYS_PROMPT.format())
        output_json = json.loads(clean_to_json_if_needed(result))
        stat_list = output_json["components"]
        if len(stat_list) == 0:
            raise EmptyOutputError(
                (
                    "No decomposition found for client statistic using supported component statistics,"
                    " cannot figure out how to calculate the requested value!"
                )
            )
    calculation = output_json["calculation"]
    added_timespan = output_json["extra_timespan"]

    add_real_time_prices = False
    today = datetime.date.today()
    if (
        stat_list == ["Close Price"]
        and (  # we are looking for up-to-date data
            args.date_range is None
            or args.date_range.end_date is None
            or args.date_range.end_date == today
        )
        and end_date == today - ONE_DAY  # but we only have yesterdays
    ):
        logger.info("Getting intraday prices to supplement Close Prices")
        real_time_prices, _, _ = await get_intraday_prices(stocks)
        rt_stock_column, rt_price_column = real_time_prices.columns
        if 0 in rt_price_column.data:
            missing_stocks = []
            for stock, price in zip(rt_stock_column.data, rt_price_column.data):
                if price == 0:
                    missing_stocks.append(stock.symbol)  # type: ignore
            logger.info(f"Real time price data for {missing_stocks[:5]} is missing, skipping")

        else:
            add_real_time_prices = True
            next_start_date = start_date + ONE_DAY
            calculation = calculation.replace(  # shift calculation forward given new data
                start_date.isoformat(), next_start_date.isoformat()
            ).replace(end_date.isoformat(), (end_date + ONE_DAY).isoformat())
            start_date = next_start_date

            eastern_time = get_now_utc().astimezone(EASTERN_TZ)

            await tool_log(
                log=(
                    f"Using intraday prices, prices retrieved at {eastern_time.time().isoformat()[:5]} Eastern"
                    f", {eastern_time.date().isoformat()}"
                ),
                context=context,
            )

    await tool_log(
        log=(
            f"Analyzed reference to '{stat_ref}'; Component variable(s): {', '.join(stat_list)}"
            f"{'; Calculation: ' + calculation if calculation else ''}"
        ),
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
        if "estimate" not in source.lower() and "quarterly" not in source.lower():
            force_daily = True

    # want daily data if looking for a single recent date
    if start_date == end_date and today - 5 * ONE_DAY <= start_date <= today + ONE_DAY:
        force_daily = True

    comp_tables = []
    for stat in stat_list:
        stat_id = StatisticId(stat_id=all_statistic_lookup[stat].feature_id, stat_name=stat)
        source = all_statistic_lookup[stat].source
        comp_tables.append(
            await get_statistic_data(
                context=context,
                stock_ids=stocks,
                statistic_id=stat_id,
                start_date=comp_start_date,
                end_date=end_date,
                force_daily=force_daily,
                add_quarterly="estimate" in source.lower() or "quarterly" in source.lower(),
                target_currency=args.target_currency,
            )
        )

    if len(comp_tables) > 1:
        comp_table: StockTable = await join_tables(  # type: ignore
            JoinTableArgs(input_tables=comp_tables), context  # type: ignore
        )
    else:
        comp_table = comp_tables[0]

    if add_real_time_prices:
        extra_dates = [today] * len(rt_stock_column.data)
        for column in comp_table.columns:  # add the extra data to the table
            if column.metadata.label == "Security":
                column.data.extend(rt_stock_column.data)
            elif column.metadata.label == "Date":
                column.data.extend(extra_dates)
            elif column.metadata.label == "Close Price":
                column.data.extend(rt_price_column.data)

        logger.info("Added real time price data")

    if not calculation and not added_timespan:
        if not is_timeseries:
            comp_table.delete_date_column()
        logger.info("No calculation explanation, returning component table")
        return comp_table

    await tool_log("Component statistic data fetched, doing calculation", context=context)

    if not is_timeseries:
        # Inject some extra instructions for the table transform.
        calculation += (
            " The output should NOT be a timeseries, and should not contain a date column. "
            "Only return the most up-to-date data for each stock."
        )

    transformed_table: StockTable = await transform_table(  # type: ignore
        args=TransformTableArgs(input_table=comp_table, transformation_description=calculation),
        context=context,
    )
    if (
        added_timespan
    ):  # if we did a rolling calculation, the extra days are usually there, remove them
        logger.info("Deleting extra days in table due to rolling calculation")
        transformed_table.delete_data_before_start_date(start_date=start_date)

    if not is_timeseries:  # just in case GPT didn't listen about deleting
        transformed_table.delete_date_column()
    transformed_table.should_subsample_large_table = True
    return transformed_table


class MacroFeatureDataInput(ToolArgs):
    statistic_reference: str
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
        " If the user does not mention any date or time frame, you should assume they "
        " want the most recent datapoint and call without specifying either start_date or end_date."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
)
async def get_macro_statistic_data(args: MacroFeatureDataInput, context: PlanRunContext) -> Table:
    if args.date_range:
        start_date, end_date = args.date_range.start_date, args.date_range.end_date
    else:
        latest_date = get_latest_date()
        end_date = latest_date
        start_date = latest_date

    # Just a sample look up
    # TODO: Add LLM decomposition to this too? Probably much less commmon
    stat_id: StatisticId = await statistic_identifier_lookup(  # type:ignore
        args=StatisticsIdentifierLookupInput(statistic_name=args.statistic_reference),
        context=context,
    )
    return await get_statistic_data(context, stat_id, start_date, end_date)


class BeatOrMissEarningsFilterInput(ToolArgs):
    stocks: List[StockID]
    miss: bool = False
    quarters: Optional[DateRange] = None


@tool(
    description=(
        "This function filters the provided stock list to only those stocks which have beat earnings"
        " expectations (EPS) or, alternatively, only those which have missed earnings expectations if "
        " miss is set to true."
        " If no quarters are provided as a date range, the stocks included in the list are chosen"
        " based on most recent quarter that where earnings data has been released. Otherwise, if one"
        " or more quarters is provided the stock must beat expecations in all of those quarters (or miss"
        " them if miss is True)."
        " The stock list returned will include, actual and expected EPS and surprise for"
        " all stocks which passed the filter, which will be displayed to the user if the stock list is"
        " printed. You should use this tool whenever a user asks to filter stocks based on whether they"
        " beat or missed earnings expectations, in this cases you must not use either the get_statistics"
        " tool, nor the transform table tool, and you should NEVER attempt to accomplish this by summarizing"
        " earnings reports, since this do not always mention relevant earnings expectations."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def beat_or_miss_earnings_filter(
    args: BeatOrMissEarningsFilterInput, context: PlanRunContext
) -> List[StockID]:
    today = datetime.date.today()
    check_most_recent = False
    if args.quarters is None or args.quarters.end_date is None:
        end_date = today
    else:
        end_date = args.quarters.end_date
    if args.quarters is None or args.quarters.start_date is None:
        start_date = end_date - TWO_QUARTERS
        check_most_recent = True
    else:
        start_date = args.quarters.start_date

    all_statistic_lookup = await get_statistic_lookup(context)

    actual_EPS = await get_statistic_data(
        context=context,
        stock_ids=args.stocks,
        statistic_id=StatisticId(
            stat_id=all_statistic_lookup[ACTUAL_EPS].feature_id, stat_name=ACTUAL_EPS
        ),
        start_date=start_date,
        end_date=end_date,
        force_daily=False,
        add_quarterly=True,
    )

    expected_EPS = await get_statistic_data(
        context=context,
        stock_ids=args.stocks,
        statistic_id=StatisticId(
            stat_id=all_statistic_lookup[EXPECTED_EPS].feature_id, stat_name=EXPECTED_EPS
        ),
        start_date=start_date,
        end_date=end_date,
        force_daily=False,
        add_quarterly=True,
    )

    per_stock_actuals = get_per_stock_data(actual_EPS)
    per_stock_expected = get_per_stock_data(expected_EPS)

    filtered_stocks = []

    for stock in per_stock_actuals:
        if stock not in per_stock_expected:
            continue

        to_check_list = []

        if check_most_recent:
            most_recent_quarter = max(per_stock_actuals[stock])
            if stock in per_stock_expected and most_recent_quarter in per_stock_expected[stock]:
                to_check_list.append(
                    (
                        most_recent_quarter,
                        per_stock_actuals[stock][most_recent_quarter],
                        per_stock_expected[stock][most_recent_quarter],
                    )
                )
        else:
            for quarter in per_stock_actuals[stock]:
                if stock in per_stock_expected and quarter in per_stock_expected[stock]:
                    to_check_list.append(
                        (
                            quarter,
                            per_stock_actuals[stock][quarter],
                            per_stock_expected[stock][quarter],
                        )
                    )

        if to_check_list and all(
            [(actual >= expected) == (not args.miss) for _, actual, expected in to_check_list]
        ):
            for quarter, actual, expected in to_check_list:
                stock = stock.inject_history_entry(
                    HistoryEntry(
                        title=f"{ACTUAL_EPS} ({quarter})",
                        explanation=actual,
                        entry_type=TableColumnType.CURRENCY,
                        unit="USD",
                    )
                )
                stock = stock.inject_history_entry(
                    HistoryEntry(
                        title=f"{EXPECTED_EPS} ({quarter})",
                        explanation=expected,
                        entry_type=TableColumnType.CURRENCY,
                        unit="USD",
                    )
                )
                stock = stock.inject_history_entry(
                    HistoryEntry(
                        title="Surprise",
                        explanation=actual / expected - 1,
                        entry_type=TableColumnType.PERCENT,
                    )
                )
            filtered_stocks.append(stock)

    if len(filtered_stocks) == 0:
        raise EmptyOutputError("No stocks passed beat/miss filter")

    # sort by surprise
    filtered_stocks.sort(
        key=lambda x: x.history[-1].explanation, reverse=not args.miss  # type:ignore
    )

    return filtered_stocks


class GetExpectedRevenueGrowth(ToolArgs):
    stocks: List[StockID]
    num_quarters: int = 4


@tool(
    description=(
        " This function creates a StockTable consisting of an expected percentage revenue growth for the"
        " provided stocks. The growth is calculated based on the difference between the actual revenue for the"
        " most recent `num_quarters` quarter where actual data exists, and the expected revenue for the following"
        "`num_quarters` quarters which do not yet have data."
        " This tool should be used together with the transform_table tool for expected revenue growth filtering, but"
        " it must not be used to identify 'growth' stocks, which is a different concept, use the growth filter"
        " for that. You must also not use this tool if the client is interested in actual past revenue growth, "
        " instead you should get the past revenue using the get statistic tool, and then calculate the desired "
        " growth number by tranforming that data."
        " But if the user wishes to compare actual and expected revenue, you must use this tool."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
)
async def get_expected_revenue_growth(
    args: GetExpectedRevenueGrowth, context: PlanRunContext
) -> StockTable:
    today = datetime.date.today()
    start_date = today
    for _ in range(args.num_quarters + 1):
        start_date -= QUARTER
    end_date = today
    for _ in range(args.num_quarters + 1):
        end_date += QUARTER

    all_statistic_lookup = await get_statistic_lookup(context)

    actual_revenue = await get_statistic_data(
        context=context,
        stock_ids=args.stocks,
        statistic_id=StatisticId(
            stat_id=all_statistic_lookup[ACTUAL_REVENUE].feature_id, stat_name=ACTUAL_REVENUE
        ),
        start_date=start_date,
        end_date=today,
        force_daily=False,
        add_quarterly=True,
    )

    expected_revenue = await get_statistic_data(
        context=context,
        stock_ids=args.stocks,
        statistic_id=StatisticId(
            stat_id=all_statistic_lookup[EXPECTED_REVENUE].feature_id, stat_name=EXPECTED_REVENUE
        ),
        start_date=today
        - QUARTER,  # back one quarter in case previous quarter data not yet released
        end_date=end_date,
        force_daily=False,
        add_quarterly=True,
    )

    per_stock_actuals = get_per_stock_data(actual_revenue)
    per_stock_expected = get_per_stock_data(expected_revenue)
    growth_lookup = {}
    past_quarter_lookup = defaultdict(list)
    future_quarter_lookup = defaultdict(list)
    past_revenue_totals = {}
    future_revenue_totals = {}
    failed_list = []
    for stock in args.stocks:
        if (
            stock in per_stock_actuals
            and stock in per_stock_expected
            and len(per_stock_actuals[stock]) >= args.num_quarters
            and len(per_stock_expected[stock]) >= args.num_quarters
        ):
            failed = False
            past_revenue_total = 0.0
            future_revenue_total = 0.0
            actuals = per_stock_actuals[stock]
            expecteds = per_stock_expected[stock]
            last_quarter_with_data = max(actuals)
            curr_quarter = last_quarter_with_data
            for _ in range(args.num_quarters):
                try:
                    past_revenue_total += actuals[curr_quarter]
                    past_quarter_lookup[stock].append(curr_quarter)
                except KeyError:
                    failed = True
                    break
                curr_quarter = get_prev_quarter(curr_quarter)
            curr_quarter = last_quarter_with_data
            future_revenue_total = 0
            for _ in range(args.num_quarters):
                curr_quarter = get_next_quarter(curr_quarter)
                try:
                    future_revenue_total += expecteds[curr_quarter]
                    future_quarter_lookup[stock].append(curr_quarter)
                except KeyError:
                    failed = True
                    break
            if failed:
                failed_list.append(stock)
                continue
            growth_lookup[stock] = future_revenue_total / past_revenue_total - 1
            past_revenue_totals[stock] = past_revenue_total
            future_revenue_totals[stock] = future_revenue_total
        else:
            failed_list.append(stock)

    if not growth_lookup:
        raise EmptyOutputError("Missing data for all stocks for projected growth calculation")
    if failed_list:
        fail_str = ", ".join([stock.company_name for stock in failed_list[:5]])
        if len(failed_list) > 5:
            fail_str += " and others"
        await tool_log(f"Missing data for {fail_str} for projected growth calculation", context)

    stocks = list(growth_lookup.keys())
    growths = [growth_lookup[stock] for stock in stocks]
    past_revenues = [past_revenue_totals[stock] for stock in stocks]
    future_revenues = [future_revenue_totals[stock] for stock in stocks]
    if args.num_quarters == 1:
        past_range = [past_quarter_lookup[stock][0] for stock in stocks]
        future_range = [future_quarter_lookup[stock][0] for stock in stocks]
    else:
        past_range = [
            f"{min(past_quarter_lookup[stock])} -- {max(past_quarter_lookup[stock])}"
            for stock in stocks
        ]
        future_range = [
            f"{min(future_quarter_lookup[stock])} -- {max(future_quarter_lookup[stock])}"
            for stock in stocks
        ]

    return StockTable(
        columns=[
            StockTableColumn(
                data=stocks,
            ),
            TableColumn(
                metadata=TableColumnMetadata(
                    label="Projected Revenue Growth", col_type=TableColumnType.PERCENT
                ),
                data=growths,  # type:ignore
            ),
            TableColumn(
                metadata=TableColumnMetadata(
                    label="Actual Revenue", col_type=TableColumnType.CURRENCY, unit="USD"
                ),
                data=past_revenues,  # type:ignore
            ),
            TableColumn(
                metadata=TableColumnMetadata(
                    label="Actual Revenue Quarter(s)", col_type=TableColumnType.STRING
                ),
                data=past_range,  # type:ignore
            ),
            TableColumn(
                metadata=TableColumnMetadata(
                    label="Projected Revenue", col_type=TableColumnType.CURRENCY, unit="USD"
                ),
                data=future_revenues,  # type:ignore
            ),
            TableColumn(
                metadata=TableColumnMetadata(
                    label="Projected Revenue Quarter(s)", col_type=TableColumnType.STRING
                ),
                data=future_range,  # type:ignore
            ),
        ]
    )


def get_per_stock_data(table: StockTable) -> Dict[StockID, Dict[str, float]]:

    for column in table.columns:
        if column.metadata.col_type == TableColumnType.STOCK:
            stock_column = column
        elif column.metadata.col_type == TableColumnType.QUARTER:
            quarter_column = column
        elif column.metadata.col_type == TableColumnType.CURRENCY:
            currency_column = column

    output_dict: Dict[StockID, Dict[str, float]] = defaultdict(dict)
    for stock, quarter, currency in zip(
        stock_column.data, quarter_column.data, currency_column.data
    ):
        if currency is not None:
            output_dict[stock][quarter] = currency  # type:ignore
    return output_dict


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
                date_range=DateRange(start_date=start_date, end_date=end_date),
            ),
            context=plan_context,
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
