import asyncio
import datetime
import inspect
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import pytz

from agent_service.external.feature_svc_client import (
    get_all_variables_metadata,
    get_intraday_prices,
)
from agent_service.GPT.constants import GPT4_O, NO_PROMPT
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
from agent_service.planner.errors import EmptyInputError, EmptyOutputError
from agent_service.tool import (
    TOOL_DEBUG_INFO,
    ToolArgs,
    ToolCategory,
    default_tool_registry,
    tool,
)
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
from agent_service.utils.pagerduty import pager_wrapper
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed
from agent_service.utils.tool_diff import get_prev_run_info

TRADING_TO_CALENDAR_RATIO = 8 / 5

DEFAULT_TIME_DELTA = datetime.timedelta(days=365)
ONE_DAY = datetime.timedelta(days=1)
QUARTER = datetime.timedelta(days=92)

EASTERN_TZ = pytz.timezone("US/Eastern")

REMOVE_STATS = set(
    ["Analyst Expectations", "Earnings Growth", "Net Income (GAAP) Actual", "EPS (GAAP) Actual"]
)

# These EPS stats don't work for all companies, so need to be fixed
ACTUAL_EPS = "EPS Normalized Actual"
EXPECTED_EPS = "EPS Normalized Consensus Median"

ACTUAL_REVENUE = "Revenue Actual"
EXPECTED_REVENUE = "Revenue Consensus Median"

DECOMPOSITION_SYS_PROMPT_STR = f"You are a financial analyst who is trying to prepare data to satisfy a client's particular information need. You will be given a reference to a statistic that the user needs, the larger conversational context in which this statistic was mentioned, and a large collection of basic statistics which your database has time series data for. Your goal is to translate this client statistic reference into one or more of the provided basic statistics and, if necessary, an explanation of any mathematical operations required to translate the selected statistic(s) into the statistic that will satisfy the client. Specifically, you will output a json with three keys. The first key should be called `components` and should be a json list of strings corresponding directly to statistics in the provided list that are either are, or are useful for calculating the required client-desired statistic. If the statistic that is being asked for is directly available in your list of statistics, and it clearly all you need to fully satisfy the client need, do not make your derivation needlessly complicated, you will be fired if you calculate a statistic you already have. Note that the client will not always use exactly the same wording, for instance they will talk about 'divided yield' but the corresponding listed statistic is Common Dividend Yield. Still, it is preferable to pick basic statistics that exactly match statistics mentioned in the input, when possible (e.g. if the client asks for ratio like price to earnings, you would choose price to earnings directly from the list rather than stock price and earnings separately). You should always search the full list carefully for an exact match before settling on a near match. Never, ever make matches to statistics that involve parentheses, e.g. 'Net Income (Excl. Excep & GW)', unless the contents of the parentheses are specifically mentioned by your client; otherwise statistics with parentheses should be your last choice, you must look very carefully for a version without parentheses! if you pick a near match when an exact match exists, you will be fired! You should never, ever need multiple statistics like X High, X Low, and X Median unless they are explicitly asked for (they are NOT useful for delta calculations, which should involve calculation over a time series of a single statistic); just pick one! Note that if the client asks for 'estimates' or 'expectations' for a standard financial statistic, you should look for a 'consensus mean' for that statistic in the list, e.g. for `analyst expectations of EBIT` you would select `EBIT Consensus Mean`. If there is no other context, analyst expectations should be interpreted as 'Target Price Consensus Mean. However, if the client does NOT ask for an estimate or expectation, you must NEVER choose a statistic with Consensus in the name. If there appears to be absolutely no way to get the desired statistic from the provided basic statistics, components should be an empty list. Otherwise, the value of the `components` key must consist of statistics that appear in the provided list, if you output a statistic that is not in the list you will be fired.  The second key, `calculation` will be an empty string if you have already found the exact statistic you need, otherwise it will be a description of any mathematical operations that must be applied to the component statistics to get the desired statistic (either an individual value for each stock, or a time series, depending on is_timeseries). This description should be clear and specific and directly refer to any statistics mentioned in the first line. If the client need is a time series, the calculation must transform the input time series into an output time series, you must NOT do a calculation which makes sense only for the last date of the time series, turning the time series into a single point! For example, if you are calculating a price change graph, you must calculate a price change for every day of the time series! To remind you of this, you MUST explicitly say in your calculation that you are generating a time series between the two provided dates, you must use the terms `generating` and `time series` and state the two dates. However, you must not output the two provided dates which is_time_series is True if the client is specifically asking for monthly, quarterly, or yearly output (e.g. show me revenue growth for each of the last 10 years). In this case, you must round the provided dates to months/quarters/years that exclude the current one, and write those instead. For example, if it is 2024-10-7 and the client wants year over year revenue growth for the last 5 years, you would state that you are outputting a time series from 2019 to 2023, but if they want monthly stock price growth for the last 6 months, you would state you are outputting a time series from 2024-04 to 2024-09 (do not include days!). For yearly, monthly, or quarterly time series, your time series must NEVER include the current month, even if today is one of the very last days of the month, i.e. if today is 2024-12-31 the final month in a monthly range must 2024-11, or the final year in a yearly range must be 2023! And because you are shifting the range back, you must also include extra_timeseries when you do this, see below. Again, for some needs, including ranking, filtering tasks, a timeseries is NOT required. In these cases you should NOT use the terms daily, weekly, or monthly, and you should only output a single datapoint per stock! Mention in your calculation description that you are outputing a single datapoint for the provided date. Always explicitly indicate any relevant date in your calculation, although you should mention ranges like week/month/quarter year, you must also mention the exact dates at each end of any time span. As much as possible, you should pass the client's original wording to the calculation function; if the input uses `over the last week` or `month`, you MUST output `over the last week` or `month` in your calculation instructions. It is critical, in particular, that your reference to the output variable in the calculation uses exactly the same wording as the client did (e.g., if the client said `performance`, your calculation description must indicate that you are calculating `performance`, and not only refer to your interpretation of performance). If you listed only one statistic in the `components` key and you are fairly confident that that statistic is in fact exactly what the client wants, you must output an empty string for the `calculation` key, however you should be very confident that the client will be satisfied with this result. Remember you must provide ONLY the statistic that is asked for; sometimes the chat content will involve a larger statistic calculation, but you must ignore that larger calculation if it is not directly relevant for producing the statistic asked for. You are always doing a calculation for each stock, so, unless you decide to include no description at all, you must use the words 'for each stock' in your description.  For the third key, `extra_timespan`, you should consider whether you need to retrieve additional historical time series data for the statistic(s) listed under components to calculate the relevant client-desired statistic. For example, to calculate a 1M moving average, you need 1M of additional data (the first datapoint in the time series requires 1M of data before it), and to calculate a delta for one year (e.g performance gain over the last year) you will need an extra year of data. If the client is asking for a single data point output but the required calculation requires more dates than that, you must always include an extra_timespan that covers all relevant dates! The timespan needed will often (though not always) be mentioned directly in the statistic. It is fine to include a bigger timespan than need but if you fall to add the extra dates you will be unable to do the calculation, so it is always better to include more extra_timespan than less.  One case where you MUST add an extra_timespan when the client needs performance number(s) for a specific month/year/multiyear span; in such a case, you MUST include extra timespan corresponding to the span of time indicated in the description. For example, if the user is looking for the performance for a specific year, you MUST include 1Y of extra timespan (so you have data for both the beginning and end of the year), and if they mention revenue change for the past 3 years, you must include 3Y of extra data, otherwise you will NOT have the data to do the calculation. Similarly, you must always always add an extra year when a user asks for yearly stats, and an extra month when the client asks for monthly stats (if the description contains a phrase 'for each month' you must add a month of extra_timespan, if the description contains for the phrae 'for each year' you must add a year!). If the chat context indicates that the user will use the statistic to do a calculation involving month-over-month or year-over-year growth for multiple years, make sure you always add not one but two extra months or years so there is always enough data to do the calculation. For example, the user has asked for annual debt growth for the last 2 years, extra_timespan should be 2Y. If the statistic requires some kind of window but the client has not said so explicitly, you can select one that seems reasonable, and include it both your calcuation and extra_timespan. If you are unsure, default to including more data than you need rather than less! Express the timespan of additional data you need as a string in standard financial format, a number followed by a single letter indicating the units (e.g. 1D, 3M, 1Y), output an empty string if no such additional data is needed (which is very common, for instance you would need no such extra data for standard ratio like Price to Earnings). A few additional guidelines:\n- You should interpret stock price to mean close price\n- Any reference to stock price change (gain/loss/delta) or returns should default to a percentage price change and must be calculated as a cumulative percent change over the period, calculated relative to the first day of the period: you must be explicit in your calculation description that you are doing a cumulative percentage price change calculation, and what the first day is (if you are creating a time series, you must state the reference date independently of the mention of the output time series range, even though the first date is almost always the same!)\n-- However, if the client specifically mentions daily performance, return, or price change, that means the percentage change of close price for each day relative to the previous day, do NOT use a cumulative calculation in that case. You must never use the Daily Returns statistic nor Open Price statistic in calculations of returns (or performance), you must calculate all daily returns using the difference between Close Price on consecutive days. This is very important!\n- You must keep all percentages in decimal form; never, ever multiply them by 100!\n-When there is ambiguity, you should always interpret weeks, months, and years as spans of time relative to the present rather than a specific date/year/month. If the user says `for/over the last month/week`, that means a time ranging consisting of the thirty/seven (calendar) days prior to today and NOT since the first day of the last month/week. 'Over the last week` means the last seven days, not  You must never jump to the conclusion that the client is talking about the first/last day of a week/month/year unless that is exactly the wording used!\n- If multiple basic statistics seem like they could be used for a specific part of the calculation, you should be biased towards chosing simpler statistics (i.e. those without modifiers)\n- LTM in this context means last twelve months, whereas our balance sheet data is generally quarterly: if someone asks for an LTM/annual balance sheet statistic, you should get 12 month of data for extra timespan and sum the results, e.g. Earnings LTM is a sum of twelve months of Earnings\n- If someone asks for total return, we also want a percentage change, but when including the dividend amount you must make sure to normalize the amounts relevant to the stock price at the beginning of the period and, if the user wants a time series, use a cumulative sum of the dividend amounts (or just a regular sum if not a time series). You must explicitly mention both these steps in your calculation description. Do not do this calcuation if the client just asks for 'returns', use it only if `total returns` (or dividend-adjusted returns. etc.) is explicitly mentioned\n- If the client mentions wanting growth of some statistic over some time period, you must always calculate a percentage growth yourself from the raw statistic, do not use any statisics with 'growth' in the name\n- If the client mentions wanting a single datapoint for a statistic that requires some calculation over a time range, but does not make that time range explicit in the chat context, you should default to one year, and mention it in your calculation description and add 1Y of extra data using extra_timespan\nGrowth calculations over larger periods like years are tricky because they depend on the nature of the statistic and the underlying data; if you are calculating revenue or volume or any other data which can be sensibly summed to get annual statistic, you will sum them over the relevant periods and compare YoY. However, for many other statistics like stock price, EPS, or P/E with just reflect a current value that cannot sensibly be summed, your goal when calculating growth is to compare the change in the value over the span, i.e. subtract the current price from the price a year ago. Whenever you do a growth calculation, your growth instructions must be fully explicit about how you are doing the calculation. This is particular important, say, when you are doing calculations for multiple years, you must mention the specific mechanisms for the calculation (either summing and subtracting yearly day, say, or finding two dates to subtract the value of). You must always do this when you are doing a growth calculation! Remember that our data itself is NEVER yearly, it is always either daily (for stock exchange related data like price or volume) or quarterly (for balance sheet data like earnings).If you are doing yearly/month growth calculation, the start date should be the end of the previous year, not the beginning of the current, please state that explicitly.\n- If the user asks for EPS, you must use `{ACTUAL_EPS}` for past data, and `{EXPECTED_EPS}' for future data. I repeat, you must choose one of those two EPS statistics when the client asks for EPS unless there is an exact string match to one of the other kinds of EPS in your stil.\n- Make sure you are only calculating the provided statistic referenced, you should not include any other calculations mentioned in the chat context, in particular do not mention anything related to any filtering or ranking in your calculation description, another analyst will handle that; your task is limited to deriving the statistic mentioned by the client\n- Do not output a wrapper around your json (no ```json!)\n- Do not mention the specific stocks from the chat, your output should work for any stock\n- Be concise, unless the calculation is very complex, a single sentence of no more than 40 words is strongly preferred, you do not need a formula."  # noqa: E501

TWO_QUARTERS = datetime.timedelta(days=183)

DECOMPOSITION_MAIN_PROMPT_STR = "Identify which of the following list of statistics is, or can be used to derive the statistic referenced by the user, as understood in the larger chat context, and provide a mathematical description of how such a derivation would occur, as needed. {time_series}\n Here is the statistic you must return: {statistic_description}\nHere is the larger chat context, delimited by `---`:\n---\n{chat_context}\n---\nAnd here is the long list of statistics you have data for, also delimited by `---`:\n---\n{statistic_list}\n---\nNow output your json consisting of a list of relevant statistics, and an explanation of how to derive the client's statistic from those statistics, and, if applicable, the amount additional time series data that must be requested beyond the timespan asked for by the client (don't forget to add extra timespan for any of the many, many cases where it is absolutely essential for a successful statistic calculation!!!!!):\n"  # noqa: E501

UPDATE_DECOMPOSITION_MAIN_PROMPT_STR = "You are a financial analyst tasked with doing a periodic calculation of financial statistics. As part of that, each time you do the calculation, you generally need to update your detailed calculation description to reflect the passage of time since the last run. You will be given a short general description of the target statistic, the previous detailed calculation description (which is usually date specific), start and end dates associated with that previous calculation, and start and end dates for the current calculation. You will rewrite the detailed calculation description in its entirety, making any date changes required. In many cases, that will simply involve subbing out any appearance of the old dates with the new ones. If the calculation involves a time range, be very, very careful to perserve the span of the time range (if the old calcluation involved a span of one month/year/week, it should be the same span after). It should never be the case that you have the same date at both the start and the end of a time range. Note that sometimes the start and end dates will be the same; if this the case, sometimes there will only be one date in the calculation description, other times there will be two dates; if there are two, usually the other date is defined relative to the start/end date (e.g. the other date is a month/quarter/year before), and if so you must change the other date to preserve that relative relationship, which will be typically be mentioned in the general calculation description. Sometimes your calculation will involve months (e.g. `2024-09`) or years (e.g. `2024`) that need to be changed. Generally you will only need to change this at the beginning of a month or year. A very common situation is that the relevant year or month is actually the last complete one before the end date; generally if there are months or years involved in the calculations and the difference between old and new dates involve a change to the year or month, there will be a corresponding change to any month or year mentioned in the detailed description, but if there is no change to the month or year, no change is required. For example, if the old end date was 2024-12-31, and the old description had a range of months such as 2024-9 to 2024-11, then, if the new end date is 2025-01-01, you would want to update the new range to be 2024-10 to 2024-12. Or, if the old description had a year range 2021 to 2023 instead, you would update that to 2022 to 2024 when changing to 2025-01-01. However, if the end date changed from 2023-12-30 to 2023-12-31, you would make no change to the description for these two cases, since the month/year has not changed. I repeat, for monthly ranges you should only make a change at the first day of a new month, and for years only make a new change in the first day of a new year. Most days you will not be making a change. Make sure you do not change anything that is not related to dates. Here is the general description of the target statistic:{statistic_reference}. Here is the detailed calculation description which you are updating: `{decomp_description}` Here is the old start date: {old_start_date}. Here is the new start date: {new_start_date}. Here is the old end date: {old_end_date}. Here is the new end date: {new_end_date}. Now rewrite the detailed description with any changes needed based on the updated date(s):\n"  # noqa: E501

DECOMPOSITION_SYS_PROMPT = Prompt(
    name="DECOMPOSITION_SYS_PROMPT",
    template=DECOMPOSITION_SYS_PROMPT_STR,
)

DECOMPOSITION_MAIN_PROMPT = Prompt(
    name="DECOMPOSITION_MAIN_PROMPT",
    template=DECOMPOSITION_MAIN_PROMPT_STR,
)

UPDATE_DECOMPOSITION_MAIN_PROMPT = Prompt(
    name="UPDATE_DECOMPOSITION_MAIN_PROMPT", template=UPDATE_DECOMPOSITION_MAIN_PROMPT_STR
)

TIME_SERIES_TEMPLATE = (
    "Note that you are generating a time series between {start_date} and {end_date} for each stock."
)

SINGLE_DATE_OTHER_TEMPLATE = (
    "Note that you are generating a single point of data for {date} for each stock. Note that date is in "
    "the past, the current date is {today}. You should interpret any relative date expression of the client's "
    "using today, but make sure you mention in your calculation that you are generating a statistic for {date}, "
    "not today! If you are doing a calculation involving more than one date in your calculation, {date} will "
    " typically be the last relevant date. "
    "If a period of time or earlier date is mentioned in the statistic description, make sure you include corresponding "
    "amount of extra_timespan!"
)

SINGLE_DATE_TODAY_TEMPLATE = (
    "Note that you are generating a single point of data for today, {date}, for each stock. If you are doing "
    "a calculation involving more than one date in your calculation, {date} will typically be the last relevant date. "
    "If a period of time or earlier date is mentioned in the statistic description, make sure you include corresponding "
    "amount of extra_timespan!"
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
        "This function returns a time series of data (sometimes one point) for a client-provided statistic"
        " for each stock in the list of stock_ids."
        " Note that the stock_ids passed into this tool must be a List[StockID], do NOT pass in an existing"
        " StockTable. If you want to add new company statistic columns to an existing table, extract the"
        " stocks with the get_stock_identifier_list_from_table tool, pass that list of stock ids to this tool"
        " to get a new table, and then join the two tables."
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
        " if the user needs growth or percentage change for some filter or ranking operation, you must do the"
        " calculation in this tool, not in a separate tool. Do as much as you can in this tool as long as it does"
        " not require some kind of filtering, ranking, or agglomeration across stocks. For example, if you are asked"
        " for top 5 stocks by P/E growth over the last year, you will pass 'P/E growth over the last year' as your"
        " statistic reference, and do ONLY the filtering for the top 5 stocks within a separate transform_table call."
        " Note that if the client asks for a statistic for a time in the past, you must both provide a date_range"
        " corresponding to that past time AND you must mention the relevant time period in the statistic description."
        " For example, if the client asks for 2021 performance of a stock, you would include a date_range for 2021 and"
        " the statistic_reference `Performance for 2021`. If the client mentions an absolute date range at all, you"
        " must assume it is in the past, and create a date range for it, even when is_time_series=False and you are"
        " mentioning it in the statistic_reference. This is very important! if a date_range is not passed in to this"
        " tool in these situations, your calculation will fail and you will be fired!"
        " Statistics that involve averaging or other amalgamation of base statistics over time should also be passed "
        " into this function, e.g. `average P/E over the last 3 years` is also a perfectly valid statistic_reference."
        " In such cases there is no time series and the date_range should be today (default)."
        " You must use this tool directly if the client is asking for a time series that includes output for each of"
        " a set of months or years, E.g. `show me revenue growth for APPL for each year of the last 10 years`."
        " In such a case, the statistic_reference MUST contain information about exact time period involved"
        " and since the output is a time series you must use is_time_series=True!"
        " The client will usually indicate they want this by either saying monthly or yearly, or they will say"
        " 'for each of N years/months' or 'every year for the last N year/months' or something similar. If the"
        " client says something like that, read the following instructions carefully!"
        " In these cases you must ALSO include a date_range, generally it is very rare to be missing a date_range"
        " if is_time_series=True. That is, for monthy/yearly statistics, (including but not limited to YoY and MoM) "
        " you must both pass a date range AND you must explicitly indicate the overall time range (e.g. 5 years,"
        " 6 months) in the statistic_reference. Both are required in this circumstance."
        " For example, if the query asks for monthly return for the last six months, you will create a date_range of"
        " six months, and pass a statistic_reference such as `monthly returns for each of the last six months`."
        " Do not forget either the date range or the reference to the date range in the statistic_reference when"
        " working with monthy or yearly statistics, if you forget either one, the calculation will not work and"
        " you will be fired!"
        " You should never, ever use the transform table tool directly to calculate"
        " monthly or year statistics (including growth statistics) for a set of stocks, even if you have already"
        " retrieved the relevant data. If you do so, your calculation will fail and you will be fired!"
        " Note you can only calculate one growth statistic at a time with this tool, you must calculate multiple"
        " statistics in multiple calls to the tool."
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
        " If the client is asking for month over month or year over year change, then they do not want a single value"
        " for each stock, and is_time_series must be set to True."
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
        " The output of this tool is always a table, and you must always label the output variable with the `_table`"
        " suffix. The output of this tool cannot be passed to a summarize tool, if you want to read the table output"
        " of this tool and produce text, you must output the table and use the analyze_outputs tool."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=default_tool_registry(),
    retries=1,
    enabled=True,
)
async def get_statistic_data_for_companies(
    args: GetStatisticDataForCompaniesInput, context: PlanRunContext
) -> StockTable:
    if context.chat is None:  # for mypy
        raise Exception("No chat context provided")

    if not args.stock_ids:
        raise EmptyInputError("No stocks to derive statistics for")

    logger = get_prefect_logger(__name__)

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)

    stat_ref = args.statistic_reference
    stocks = args.stock_ids

    today = get_now_utc().date()
    # do the min in case we are doing this in the past
    latest_date = min(get_latest_date(), today)

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
            await tool_log(
                log=(
                    "No date range provided for time series output, "
                    f"defaulting to {start_date.isoformat()} to {end_date.isoformat()}"
                ),
                context=context,
            )
        else:
            end_date = latest_date
            start_date = latest_date

    decomp_json = {}

    all_statistic_lookup = await get_statistic_lookup(context)

    no_cache = True

    try:  # since everything here is optional, put in try/except
        # TODO: maybe update so we don't have to pull in the input or output when we don't need it?
        prev_run_info = await get_prev_run_info(context, "get_statistic_data_for_companies")
        if prev_run_info is not None:
            prev_other: Dict[str, str] = prev_run_info.debug  # type:ignore

            if prev_other:
                old_start_date = prev_other["start_date"]
                old_end_date = prev_other["end_date"]
                old_decomp = json.loads(prev_other["decomp_json"])
                old_decomp_description = old_decomp["calculation"]

                # When we disable old statistics, we need to force a reset of any calculations
                # that use it

                bad_stat = False

                for stat in old_decomp["components"]:
                    if stat not in all_statistic_lookup:
                        bad_stat = stat

                if not bad_stat:
                    if old_decomp_description == "" or (
                        start_date.isoformat() == old_start_date
                        and end_date.isoformat() == old_end_date
                    ):
                        new_calculation = old_decomp_description
                    else:
                        update_main_prompt = UPDATE_DECOMPOSITION_MAIN_PROMPT.format(
                            statistic_reference=args.statistic_reference,
                            decomp_description=old_decomp_description,
                            old_start_date=old_start_date,
                            old_end_date=old_end_date,
                            new_start_date=start_date,
                            new_end_date=end_date,
                        )

                        new_calculation = await llm.do_chat_w_sys_prompt(
                            update_main_prompt, NO_PROMPT
                        )

                    decomp_json = {
                        "extra_timespan": old_decomp["extra_timespan"],
                        "components": old_decomp["components"],
                        "calculation": new_calculation,
                    }

                    await tool_log(
                        log=("Loaded statistic calculation description from previous run"),
                        context=context,
                    )
                    no_cache = False
                else:
                    logger.warning(f"Failed to load statistic: {bad_stat}, redoing from scratch")

    except Exception as e:
        logger.exception(f"Error creating diff info from previous run: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to get previous run info or getting default stock list",
        )

    debug_info: Dict[str, Any] = {}
    TOOL_DEBUG_INFO.set(debug_info)

    if not decomp_json:
        # sorting by length encourages it to pick simpler stats (seems to start looking at top)
        all_statistics = "\n".join(sorted(all_statistic_lookup, key=lambda x: len(x)))
        if is_timeseries:
            time_series_str = TIME_SERIES_TEMPLATE.format(start_date=start_date, end_date=end_date)
        elif (
            not args.date_range or not args.date_range.end_date or today == args.date_range.end_date
        ):
            time_series_str = SINGLE_DATE_TODAY_TEMPLATE.format(date=end_date)
        else:
            time_series_str = SINGLE_DATE_OTHER_TEMPLATE.format(date=end_date, today=today)

        main_prompt = DECOMPOSITION_MAIN_PROMPT.format(
            statistic_description=stat_ref,
            chat_context=context.chat.get_gpt_input(),
            statistic_list=all_statistics,
            time_series=time_series_str,
        )
        result = await llm.do_chat_w_sys_prompt(main_prompt, DECOMPOSITION_SYS_PROMPT.format())
        decomp_json = json.loads(clean_to_json_if_needed(result))
        stat_list = decomp_json["components"]
        if len(stat_list) == 0:
            main_prompt = DECOMPOSITION_MAIN_PROMPT.format(
                statistic_description=stat_ref,
                chat_context=stat_ref,
                statistic_list=all_statistics,
                time_series=time_series_str,
            )
            result = await llm.do_chat_w_sys_prompt(main_prompt, DECOMPOSITION_SYS_PROMPT.format())
            decomp_json = json.loads(clean_to_json_if_needed(result))
            stat_list = decomp_json["components"]
            if any([stat not in all_statistic_lookup for stat in stat_list]):
                logger.warning(
                    f"One of statistics in {stat_list} is not a supported stat, retrying"
                )
                result = await llm.do_chat_w_sys_prompt(
                    main_prompt, DECOMPOSITION_SYS_PROMPT.format()
                )
                decomp_json = json.loads(clean_to_json_if_needed(result))
                stat_list = decomp_json["components"]
            if len(stat_list) == 0 or any([stat not in all_statistic_lookup for stat in stat_list]):
                raise EmptyOutputError(
                    (
                        f"No decomposition found for client statistic '{stat_ref}'"
                        " using supported component statistics,"
                        " cannot figure out how to calculate the requested value!"
                    )
                )

    debug_info["decomp_json"] = json.dumps(decomp_json)
    debug_info["start_date"] = start_date.isoformat()
    debug_info["end_date"] = end_date.isoformat()

    stat_list = decomp_json["components"]
    calculation = decomp_json["calculation"]
    added_timespan = decomp_json["extra_timespan"]

    add_real_time_prices = False
    rt_stock_column = None
    rt_price_column = None
    if (
        stat_list == ["Close Price"]
        and (  # we are looking for up-to-date data
            args.date_range is None
            or args.date_range.end_date is None
            or args.date_range.end_date == today
        )
        and (
            end_date == today - ONE_DAY
            or (end_date.weekday() == 0 and end_date == today - 3 * ONE_DAY)
        )
        # but we don't have today's prices
        # TODO: make long weekends work?
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
            next_start_date = start_date + ONE_DAY  # should be okay if this ends up in a weekend
            calculation = calculation.replace(  # shift calculation forward given new data
                start_date.isoformat(), next_start_date.isoformat()
            ).replace(end_date.isoformat(), today.isoformat())
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

    if (
        start_date == end_date
        and args.date_range
        and args.date_range.start_date
        and args.date_range.end_date
    ):
        # looking for single output date, but input range suggests we need need more dates
        days_from_date_range = (args.date_range.end_date - args.date_range.start_date).days
    else:
        days_from_date_range = 0

    if days_from_date_range or added_timespan:
        try:
            gpt_extra_days = convert_horizon_to_days(added_timespan)
        except Exception as e:
            logger.warning(
                f"Failed to convert GPT-derived extra timespan: {added_timespan} due to {e}"
            )
            gpt_extra_days = 0
        extra_days = max(gpt_extra_days, days_from_date_range)
        if (
            "D" in added_timespan and extra_days > 5
        ):  # D refers to trading days, need more calendar days
            extra_days = int(
                extra_days * TRADING_TO_CALENDAR_RATIO
            )  # doesn't hurt to get extra, they'll be droppped
        logger.info(f"Extra days included for calculation: {extra_days}")
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
    if (
        start_date == end_date
        and today - 5 * ONE_DAY <= start_date <= today + ONE_DAY
        and not added_timespan
    ):
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
            JoinTableArgs(input_tables=comp_tables),  # type: ignore
            context,
        )
    else:
        comp_table = comp_tables[0]

    if add_real_time_prices and rt_stock_column is not None and rt_price_column is not None:
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
        args=TransformTableArgs(
            input_table=comp_table, transformation_description=calculation, no_cache=no_cache
        ),
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
    tool_registry=default_tool_registry(),
    enabled=True,
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
    quarters: Union[Optional[DateRange], int] = None
    mode: str = "earnings"
    filter: bool = True


@tool(
    description=(
        "This function filters the provided stock list to only those stocks which have beat earnings"
        " expectations (EPS) or, alternatively, only those which have missed earnings expectations if "
        " miss is set to true."
        " if the user asks for a specific quarter or date range in the past, e.g. 2021Q2, a date range"
        " should be passed to this tool as the quarters argument. Otherwise, if the user is just"
        " interested in the last N quarters (or last N years), pass the integer number of quarters the"
        " user wants to check. Never, ever pass a date_range object when the user asks for filtering"
        " of the last N quarter/years, this will not work properly."
        " If no quarters are provided as a date range, the stocks included in the list are chosen"
        " based on most recent quarter that where earnings data has been released. Otherwise, if one"
        " or more quarters is provided the stock must beat expecations in all of those quarters (or miss"
        " them if miss is True)."
        " If quarters is an int, or not provided, the behavior of this tool is to default to the most"
        " recent reported quarters for each stock. When writing the output title when outputting"
        " the list returned by this tool using the prepare_output tool, the title should specifically"
        " mention 'Last N Reported Quarters or `Last Reported Quarter` if quarters is 1 or None. For"
        " example, if the user asked for stocks which beat expectations each quarter for the last year"
        " for prepare_output would say instead 'beat expecations for the last 4 reported quarters."
        " Usually you will want the filter argument to be True (the default), but if the user wants"
        " beat/miss info about a stock or a handful of stocks without applying any filter, you set"
        " filter = False. for example, if the user asks 'has GE met earning expectations recently?"
        " you must set filter = False, since the client wants to see info about GE whether or not it"
        " made earnings expectations! You should always set filter = False if the client asks a "
        " beat/miss question about a specific stock!"
        " Assuming filter=True, The stock list returned will include, actual and expected EPS and surprise for"
        " all stocks which passed the filter, which will be displayed to the user if the stock list is"
        " printed. You should use this tool whenever a user asks to filter stocks based on whether they"
        " beat or missed earnings expectations, in this cases you must not use either the get_statistics"
        " tool, nor the transform table tool, and you should NEVER attempt to accomplish this by summarizing"
        " earnings reports, since this do not always mention relevant earnings expectations."
        " If the user is interested in beat or miss for revenue expecations, pass revenue as the mode"
        " instead of earnings, but you should default to earnings unless revenue is specifically mentioned."
        " only 'earnings' and 'revenue' are supported as possible modes"
        " This tool is not a substitute for reading earnings calls, you must never, ever use this tool"
        " if the user is asking you to read earnings!"
        " This tool does not have access to earnings guidance, you must not use this tool the user is"
        " interested in earnings guidance, instead you must read the earnings calls where guidance is"
        " discussed. Again, do not use this tool for earnings guidance related requests!"
        " This tool must NOT be used in isolation when the client asks about beat/miss expectations"
        " for future earnings, since this only provides past data. It is more important in such"
        " situations to do a summary of news and to look at earnings expectations growth, which does"
        " related to future data. This is still useful for context in those situations, but you must"
        " never use only this tool when the user is asking about beat/miss expectations! If you do, you"
        " will be fired!"
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=default_tool_registry(),
)
async def beat_or_miss_earnings_filter(
    args: BeatOrMissEarningsFilterInput, context: PlanRunContext
) -> List[StockID]:
    today = datetime.date.today()
    check_most_recent_quarters = 0
    if args.quarters is None or isinstance(args.quarters, int) or args.quarters.end_date is None:
        end_date = today
    else:
        end_date = args.quarters.end_date
    if args.quarters is None or (
        not isinstance(args.quarters, int) and args.quarters.start_date is None
    ):
        start_date = end_date - TWO_QUARTERS
        check_most_recent_quarters = 1
    elif isinstance(args.quarters, int):
        start_date = (
            today - args.quarters * QUARTER - TWO_QUARTERS
        )  # get extra data so we get 4 quarters
        check_most_recent_quarters = args.quarters
    else:
        start_date = args.quarters.start_date

    if check_most_recent_quarters:
        await tool_log(
            log=f"Checking the last {check_most_recent_quarters} quarters with reported data",
            context=context,
        )
    else:
        await tool_log(
            log=(f"Checking quarters between {start_date.isoformat()} and {end_date.isoformat()}"),
            context=context,
        )

    all_statistic_lookup = await get_statistic_lookup(context)

    if args.mode == "earnings":
        actual_stat = ACTUAL_EPS
        expected_stat = EXPECTED_EPS
    elif args.mode == "revenue":
        actual_stat = ACTUAL_REVENUE
        expected_stat = EXPECTED_REVENUE
    else:
        raise Exception(f"unsupported {args.mode=}")

    actual_EPS = await get_statistic_data(
        context=context,
        stock_ids=args.stocks,
        statistic_id=StatisticId(
            stat_id=all_statistic_lookup[actual_stat].feature_id, stat_name=actual_stat
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
            stat_id=all_statistic_lookup[expected_stat].feature_id, stat_name=expected_stat
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

        stock_actuals = per_stock_actuals[stock]
        stock_expected = per_stock_expected[stock]

        to_check_list: List[Tuple[str, float, float]] = []

        if check_most_recent_quarters:
            if len(stock_actuals) < check_most_recent_quarters:
                continue  # has to have at least the right number of quarters
            most_recent_quarters = sorted(stock_actuals, reverse=True)
            for i, recent_quarter in enumerate(most_recent_quarters[:check_most_recent_quarters]):
                # remove stocks with noncontiguous quarters, missing data
                if i != 0 and most_recent_quarters[i - 1] != get_next_quarter(recent_quarter):
                    to_check_list = []
                    break
                if recent_quarter in stock_expected:
                    to_check_list.append(
                        (
                            recent_quarter,
                            stock_actuals[recent_quarter],
                            stock_expected[recent_quarter],
                        )
                    )
                else:
                    to_check_list = []
                    break
        else:
            # TODO: Technically a stock with some missing data over the time period could pass this
            # However since quarters for particular stocks can be very out of sync (fiscal years), very
            # difficult to stop this case without causing other stocks to potentially fail, so leaving
            # it for now
            for quarter in sorted(stock_actuals, reverse=True):
                if quarter in stock_expected:
                    to_check_list.append(
                        (
                            quarter,
                            stock_actuals[quarter],
                            stock_expected[quarter],
                        )
                    )
                else:
                    to_check_list = []
                    break

        if to_check_list and all(
            [
                ((actual >= expected) == (not args.miss) or not args.filter)
                for _, actual, expected in to_check_list
            ]
        ):
            for quarter, actual, expected in to_check_list:
                stock = stock.inject_history_entry(
                    HistoryEntry(
                        title=f"{actual_stat} ({quarter})",
                        explanation=actual,
                        entry_type=TableColumnType.CURRENCY,
                        unit="USD",
                    )
                )
                stock = stock.inject_history_entry(
                    HistoryEntry(
                        title=f"{expected_stat} ({quarter})",
                        explanation=expected,
                        entry_type=TableColumnType.CURRENCY,
                        unit="USD",
                    )
                )
                # can't calculate surprise properly if comparison number is zero or negative
                stock = stock.inject_history_entry(
                    HistoryEntry(
                        title=f"Surprise ({quarter})",
                        explanation=(actual / expected - 1) if expected > 0 else 0,
                        entry_type=TableColumnType.PERCENT,
                    )
                )
            filtered_stocks.append(stock)

    if len(filtered_stocks) == 0:
        raise EmptyOutputError("No stocks passed beat/miss filter")

    # sort first by quarter (only difference between titles is quarter), then by suprise
    multipler = -1 if args.miss else 1
    filtered_stocks.sort(
        key=lambda x: (x.history[-1].title, x.history[-1].explanation * multipler),  # type:ignore
        reverse=True,
    )

    return filtered_stocks


class GetExpectedRevenueGrowth(ToolArgs):
    stocks: List[StockID]
    num_quarters: int = 4
    mode: str = "revenue"


@tool(
    description=(
        " This function creates a StockTable consisting of an expected percentage revenue growth for the"
        " provided stocks. The growth is calculated based on the difference between the actual revenue for the"
        " most recent `num_quarters` quarter where actual data exists, and the expected revenue for the following"
        "`num_quarters` quarters which do not yet have data."
        " This tool's output include the expected EPS or revenue (depending on the mode) for the next num_quarter "
        " quarters. If the client is asking for this information, you should generally call this tool instead of the"
        " get statistics tool."
        " This tool should be used together with the transform_table tool for expected revenue growth filtering, but"
        " it must not be used to identify 'growth' stocks, which is a different concept, use the growth filter"
        " for that. You must also not use this tool if the client is interested in actual past revenue growth, "
        " instead you should get the past revenue using the get statistic tool, and then calculate the desired "
        " growth number by tranforming that data."
        " But if the user wishes to compare actual and expected revenue, you must use this tool."
        " If the user instead wants earnings growth, you may set mode='earnings' instead. Only 'revenue'"
        " and 'earnings' are supported as possible models."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=default_tool_registry(),
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

    if args.mode == "earnings":
        actual_stat = ACTUAL_EPS
        expected_stat = EXPECTED_EPS
    elif args.mode == "revenue":
        actual_stat = ACTUAL_REVENUE
        expected_stat = EXPECTED_REVENUE
    else:
        raise Exception(f"unsupported {args.mode=}")

    actual_revenue = await get_statistic_data(
        context=context,
        stock_ids=args.stocks,
        statistic_id=StatisticId(
            stat_id=all_statistic_lookup[actual_stat].feature_id, stat_name=actual_stat
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
            stat_id=all_statistic_lookup[expected_stat].feature_id, stat_name=expected_stat
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
            # can't calculate growth properly if comparison number is zero or negative
            growth_lookup[stock] = (
                (future_revenue_total / past_revenue_total - 1) if past_revenue_total > 0 else 0
            )
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
                    label=f"Projected {args.mode.title()} Growth", col_type=TableColumnType.PERCENT
                ),
                data=growths,  # type:ignore
            ),
            TableColumn(
                metadata=TableColumnMetadata(
                    label=actual_stat, col_type=TableColumnType.CURRENCY, unit="USD"
                ),
                data=past_revenues,  # type:ignore
            ),
            TableColumn(
                metadata=TableColumnMetadata(
                    label=f"{actual_stat} Quarter(s)", col_type=TableColumnType.STRING
                ),
                data=past_range,  # type:ignore
            ),
            TableColumn(
                metadata=TableColumnMetadata(
                    label=expected_stat, col_type=TableColumnType.CURRENCY, unit="USD"
                ),
                data=future_revenues,  # type:ignore
            ),
            TableColumn(
                metadata=TableColumnMetadata(
                    label=f"{expected_stat} Quarter(s)", col_type=TableColumnType.STRING
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
        stock_column.data,  # static analysis: ignore
        quarter_column.data,  # static analysis: ignore
        currency_column.data,  # static analysis: ignore
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
