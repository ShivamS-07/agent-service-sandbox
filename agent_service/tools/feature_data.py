import datetime
import functools
import json
from threading import Lock
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd
from cachetools import TTLCache, cached
from feature_service_proto_v1.feature_service_common_messages_pb2 import (
    FEATURE_VALUE_UNITS_PERCENT,
    FEATURE_VALUE_UNITS_PRICE,
    FEATURE_VALUE_UNITS_UNIT,
)
from feature_service_proto_v1.feature_service_pb2 import (
    GetFeatureDataResponse,
    TimeAxis,
)
from gbi_common_py_utils.numpy_common.numpy_cube import NumpyCube
from gbi_common_py_utils.numpy_common.numpy_sheet import NumpySheet

from agent_service.external.feature_svc_client import get_feature_data
from agent_service.GPT.constants import FILTER_CONCURRENCY, HAIKU, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.dates import DateRange
from agent_service.io_types.graph import GraphType
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    StockTable,
    Table,
    TableColumnMetadata,
    TableColumnType,
)
from agent_service.io_types.text import Text
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.iterables import chunk
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt

LATEST_DATE_SECONDS = 60 * 60
TTLCache(maxsize=1, ttl=LATEST_DATE_SECONDS)

# need to expose this type to the type checker but avoid importing it until needed normally
if TYPE_CHECKING:
    from data_access_layer.core.dao.features.features_dao import FeaturesDAO


@functools.cache
def get_FeaturesDAO() -> "FeaturesDAO":
    from data_access_layer.core.dao.features.features_dao import FeaturesDAO

    # cache globally so that metadata caches do not have to be created on each
    # new request

    _FEATURES_DAO = FeaturesDAO()
    return _FEATURES_DAO


@io_type
class StatisticId(ComplexIOBase):
    stat_id: str
    stat_name: str

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        t: Text = Text(val=f"Statistic: {self.stat_name}")
        return await t.to_rich_output(pg=pg, title=title)


STATISTIC_CONFIRMATION_PROMPT = Prompt(
    name="STATISTIC_CONFIRMATION_PROMPT",
    template="""
Your task is to determine if the search term  means the same thing as the candidate match in a finance context
search term: '{search}'
candidate match: '{match}'
If the search term means the same thing as the candidate match return "answer" : "true"
if not a good match, return "answer" : "false"
also give a short reason to explain your answer in a "reason" field
Make sure to return this in JSON.
ONLY RETURN IN JSON. DO NOT RETURN NON-JSON.
Return in this format: {{"answer":"", "reason":""}}
""",
)


async def is_statistic_correct(context: PlanRunContext, search: str, match: str) -> bool:
    logger = get_prefect_logger(__name__)
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=HAIKU)

    prompt = STATISTIC_CONFIRMATION_PROMPT.format(search=search, match=match)

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=prompt,
        sys_prompt=NO_PROMPT,
        output_json=True,
    )

    result = result.strip().lower()

    def convert_result(result: str) -> bool:
        if result.startswith("yes"):
            return True

        if result.startswith("true"):
            return True

        if not result.startswith("{"):
            return False

        try:
            res_obj = json.loads(result)
            answer = res_obj.get("answer", False)
            if str(answer).lower() in ["yes", "true"]:
                return True
        except Exception as e:
            print(f"json parse: {repr(e)} : {result=}")
        return False

    answer = convert_result(result)
    logger.info(f"'{answer=}' '{search=}', '{match=}', '{result=}',")
    return answer


class StatisticsIdentifierLookupInput(ToolArgs):
    # name of the statistic to lookup
    statistic_name: str


@tool(
    description=(
        "This function takes a string (close price, Churn low, Market Capitalization, Coppock Curve, e.g.)"
        " which refers to a financial indicator or statistic, and matches it to an identifier for that"
        " statistic. You should use this before fetching any data for the statistic. Note that for things like"
        " 'percent change of price' or 'price delta', just fetch the underlying field 'price'"
        " and compute the delta separately."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
    is_visible=False,
    enabled=False,
)
async def statistic_identifier_lookup(
    args: StatisticsIdentifierLookupInput, context: PlanRunContext
) -> StatisticId:
    """Returns the identifier of a statistic given its name (Churn low, Market Capitalization, e.g.)

    This function performs word similarity name match to find the statistic's identifier.


    Args:
        args (StatisticsIdentifierLookupInput): The input arguments for the statistic lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        str: The integer identifier of the statistic.
    """
    logger = get_prefect_logger(__name__)
    db = get_psql()
    # TODO :
    # 1. Add more filtering or new column (agent_supported) to table
    # 2. implement solar solution

    # Word similarity name match
    sql = """
    SELECT id, name, match_name, description, ws FROM (
        select
            alt_match.id, alt_match.name, alt_match.description, alt_match.match_name,
            (
                strict_word_similarity(
                    lower(alt_match.match_name),
                    lower(%(search_text)s)
                ) + strict_word_similarity(
                    lower(%(search_text)s),
                    lower(alt_match.match_name)
                )
            ) / 2.0 as ws
        from (
            select
                id, name, description,
                unnest(ARRAY_APPEND(alternate_names, name)) as match_name
            from feature_service.available_features
        ) as alt_match
        order by ws desc
        limit 20
    ) as feats
    WHERE ws > 0.2
    """
    rows = db.generic_read(sql, {"search_text": args.statistic_name})
    logger.info(f"found {len(rows)} potential matches for '{args.statistic_name}'")

    if not rows:
        raise ValueError(f"Could not find a stock data field related to: {args.statistic_name}")

    for r in rows:
        logger.info(f"searched  '{args.statistic_name}' and found potential match: {str(r)[:250]}")

    if rows[0]["ws"] > 0.9:
        row = rows[0]
        logger.info(f"searched  '{args.statistic_name}' and found best match: {str(row)[:250]}")
        await tool_log(
            log=f"Interpreting '{args.statistic_name}' as {row['name']}", context=context
        )
        return StatisticId(stat_id=row["id"], stat_name=row["name"])

    tasks = [
        is_statistic_correct(context, search=args.statistic_name, match=r["match_name"])
        for r in rows
    ]

    results = await gather_with_concurrency(tasks, n=FILTER_CONCURRENCY)

    for result, row in zip(results, rows):
        if not result:
            continue

        logger.info(f"searched  '{args.statistic_name}' and found best match: {str(row)[:250]}")
        await tool_log(
            log=f"Interpreting '{args.statistic_name}' as {row['name']}", context=context
        )

        return StatisticId(stat_id=row["id"], stat_name=row["name"])

    raise ValueError(f"Could not find a stock data field related to: {args.statistic_name}")


class MacroFeatureDataInput(ToolArgs):
    statistic_id: StatisticId
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None
    # in the future we may want to take a currency as well


@tool(
    description=(
        "This function returns the time series of data for a statistic_id"
        " that is not tied to a specific stock. These are usually macroeconomic indicators like"
        " bank interest rates, inflation and unemployment rates."
        " Optionally a start_date and end_date may be provided to specify a date range"
        " to get a specific date only then set both start_date and end_date to the same date."
        " if the optional date_range argument is passed in it will override anything set in start_date and end_date "
        " If none of start_date, end_date, date_range are provided then it will assume the request "
        " is for the most recent date for which data exists. The statistic_id MUST be "
        " fetched with the lookup function, it cannot be an arbitrary string. "
        " If the user does not mention any date or time frame, you should assume they "
        " want the most recent datapoint and call without specifying either start_date or end_date."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def get_macro_statistic_data(args: MacroFeatureDataInput, context: PlanRunContext) -> Table:
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

    return await get_statistic_data(
        context=context,
        statistic_id=args.statistic_id,
        start_date=start_date,
        end_date=end_date,
    )


class FeatureDataInput(ToolArgs):
    stock_ids: List[StockID]
    statistic_id: StatisticId
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function returns the time series of data for a statistic_id"
        " for each stock in the list of stock_ids."
        " if you need the same stats for the same time period for more than one companies, you must call this"
        " function with multiple stock_ids, DO NOT call this function multiple times"
        " with a single stock per time in those circumstances!"
        " Optionally a start_date and end_date may be provided to specify a date range"
        " to get a specific date only  then set both start_date and end_date to the same date."
        " if the optional date_range argument is passed in it will override anything set in start_date and end_date "
        " If none of start_date, end_date, date_range are provided then it will assume the request "
        " is for the most recent date for which data exists. The statistic_id MUST be "
        " fetched with the lookup function, it cannot be an arbitrary string. "
        " If the user does not mention any date or time frame, you should assume they "
        " want the most recent datapoint and call without specifying either start_date or end_date."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
    enabled=False,
)
async def get_statistic_data_for_companies(
    args: FeatureDataInput, context: PlanRunContext
) -> StockTable:
    """Returns the Time series of data for the requested field for each of the stocks_ids
    Optionally a start_date and end_date may be provided to specify a date range
    To get a specific date only set both inputs to the same date
    if only one of them is filled in or not None then it will be assumed to be a single date to be returned
    If neither date is provided then it will assume the request is for the most recent date for which data exists

    Args:
        args (FeatureDataInput): The input arguments for the feature data retrieval.
        context (PlanRunContext): The context of the plan run.

    Returns:
        Table: The requested data.
    """
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

    return await get_statistic_data(
        context=context,
        statistic_id=args.statistic_id,
        start_date=start_date,
        end_date=end_date,
        stock_ids=args.stock_ids,
    )


async def get_statistic_data(
    context: PlanRunContext,
    statistic_id: StatisticId,
    start_date: datetime.date,
    end_date: datetime.date,
    stock_ids: Optional[List[StockID]] = None,
    force_daily: bool = False,
    ffill_days: Optional[int] = None,
    add_quarterly: bool = False,
    target_currency: Optional[str] = None,
) -> StockTable:
    stock_ids = stock_ids or []
    gbi_id_map = {stock.gbi_id: stock for stock in stock_ids}
    gbi_ids = list(gbi_id_map.keys())
    # if one date given, turn on ffill.
    ffill_days_val = ffill_days if ffill_days is not None else 0

    use_natural_axis = True  # use the axis the data prefers (quarterly or data)

    if (ffill_days is None) and (force_daily or (end_date - start_date).days < 90):
        # need to turn quarterly into daily and forward fill if it is going to be combined with daily
        # or if the range is less than a quarter
        ffill_days_val = 180
        use_natural_axis = False

    logger = get_prefect_logger(__name__)
    logger.info(
        f"getting data for gbi_ids: {gbi_ids}, "
        f"features: {statistic_id}, "
        f"{start_date=}, {end_date=}"
    )

    # use target currency provided, or default to USD

    if target_currency:
        use_currency = target_currency
    else:
        use_currency = "USD"

    logger.info(f"using currency: {use_currency}")

    # TODO: is there a better way to get this across the wire? send as raw bytes?
    # Arbitrary chunk size of 300 - feel free to adjust as needed.
    # Generally, feature service is capable of easily returning data for 300 stocks
    # over large time ranges in seconds.
    df_chunks: List[pd.DataFrame] = []
    # make sure at least one chunk is sent - empty gbiid list is allowed for global
    # macro series.
    gbi_id_chunks = list(chunk(gbi_ids, n=300)) or [[]]
    for chunk_i, gbi_id_chunk in enumerate(gbi_id_chunks):
        logger.info(f"Fetching {statistic_id=} for {chunk_i + 1} / {len(gbi_id_chunks)} chunks.")
        result: GetFeatureDataResponse = await get_feature_data(
            user_id=context.user_id,
            statistic_ids=[statistic_id.stat_id],
            stock_ids=gbi_id_chunk,
            from_date=start_date,
            to_date=end_date,
            ffill_days=ffill_days_val,
            use_natural_axis=use_natural_axis,
            target_currency=use_currency,
        )

        # make the dataframe with index = dates, columns = stock ids
        # or columns = statistic id (for global features)
        # since we are requesting data for a single statistic_id, we can do a simple check
        # and slice out the relevant data
        security_data = None
        global_data = None
        data_time_axis = None
        for sec_data in result.security_data:
            np_cube = NumpyCube.initialize_from_proto_bytes(
                sec_data.data_cube, cols_are_dates=False
            )
            if statistic_id.stat_id in np_cube.row_map:
                is_global = False
                data_time_axis = sec_data.time_axis
                security_data = np_cube
                break
        for gl_data in result.global_data:
            np_sheet = NumpySheet.initialize_from_proto_bytes(
                gl_data.data_sheet, cols_are_dates=False
            )
            if statistic_id.stat_id in np_sheet.row_map:
                is_global = True
                data_time_axis = gl_data.time_axis
                global_data = np_sheet
                break
        assert data_time_axis is not None
        index_name = "Date"
        index_type = TableColumnType.DATE
        prefer_graph_type = GraphType.LINE
        if data_time_axis != TimeAxis.TIME_AXIS_DATE:
            index_name = "Period"
            index_type = TableColumnType.QUARTER
            prefer_graph_type = GraphType.BAR

        if is_global:
            assert global_data is not None
            data = global_data.np_data[global_data.row_map[statistic_id.stat_id], :]
            df_chunk = pd.DataFrame(
                data,
                index=(
                    global_data.columns
                    if index_type == TableColumnType.QUARTER
                    else pd.to_datetime(global_data.columns)
                ),
                columns=[statistic_id.stat_name],
            )
        else:
            assert security_data is not None
            if not stock_ids:
                raise ValueError("No stocks given to look up statistic for.")
            # dims are (feats, dates, secs) -> (dates, secs)
            data = security_data.np_data[security_data.row_map[statistic_id.stat_id], :, :]
            df_chunk = pd.DataFrame(
                data,
                index=(
                    security_data.columns
                    if index_type == TableColumnType.QUARTER
                    else pd.to_datetime(security_data.columns)
                ),
                columns=[int(s) for s in security_data.fields],
            )
        df_chunks.append(df_chunk)

    # at this point, the dataframe has index = dates, columns = features(global) or stocks(nonglobal)
    # concatenate the chunks along the columns
    # here, lets assert we got at least one dataframe back before concatenating chunks.
    assert len(df_chunks) > 0
    df = pd.concat(df_chunks, axis=1)
    df = df.dropna(axis="index", how="all")

    # wrangle units
    units = dict(result.feature_value_units).get(statistic_id.stat_id, FEATURE_VALUE_UNITS_UNIT)
    if units == FEATURE_VALUE_UNITS_PRICE:
        value_coltype = TableColumnType.CURRENCY
    elif units == FEATURE_VALUE_UNITS_PERCENT:
        value_coltype = TableColumnType.PERCENT
    else:
        value_coltype = TableColumnType.FLOAT

    # turn dataframe into records.
    # TODO - should global statistics be its own tool? how?
    # TODO handle smarter column types, etc.

    # we are guaranteed to get back a single currency for every security
    # because we asked the rpc to do so. just take the first.
    # coerce empty string to None (grpc serialization) - if the feature was not currency-valued
    curr_unit = (
        result.iso_currency
        if result.iso_currency and value_coltype == TableColumnType.CURRENCY
        else None
    )
    statistic_label = statistic_id.stat_name
    if add_quarterly:
        statistic_label = "Quarterly " + statistic_label
        df.rename(columns={statistic_id.stat_name: statistic_label}, inplace=True)

    if is_global:
        # if global, the stock column is actually a feature column
        df.index.rename(index_name, inplace=True)
        df.reset_index(inplace=True)

        # We now have a dataframe with only a few columns: Date, Statistic Name
        # currency may or may not be set.
        df = df[df[statistic_label].notna()]
        stock_table = StockTable.from_df_and_cols(
            data=df,
            columns=[
                TableColumnMetadata(label=index_name, col_type=index_type),
                TableColumnMetadata(
                    label=statistic_label,
                    col_type=value_coltype,
                    unit=curr_unit,
                    data_src=[statistic_id.stat_id],
                ),
            ],
        )
    else:
        # if not global, augment with currency information if exists.
        df.index.rename(index_name, inplace=True)
        df.reset_index(inplace=True)
        df = df.melt(
            id_vars=[index_name],
            var_name=STOCK_ID_COL_NAME_DEFAULT,
            value_name=statistic_label,
        )
        # Map back to the StockID objects
        df[STOCK_ID_COL_NAME_DEFAULT] = df[STOCK_ID_COL_NAME_DEFAULT].map(gbi_id_map)

        # We now have a dataframe with only a few columns: Date, Stock ID, Statistic Name
        df = df[df[statistic_label].notna()]
        stock_table = StockTable.from_df_and_cols(
            data=df,
            columns=[
                TableColumnMetadata(label=index_name, col_type=index_type),
                TableColumnMetadata(
                    label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK
                ),
                TableColumnMetadata(
                    label=statistic_label,
                    col_type=value_coltype,
                    unit=curr_unit,
                    data_src=[statistic_id.stat_id],
                ),
            ],
        )
    stock_table.prefer_graph_type = prefer_graph_type
    return stock_table


# TODO in the future we need a latest date per region
@cached(cache=TTLCache(maxsize=1, ttl=LATEST_DATE_SECONDS), lock=Lock())
def get_latest_date() -> datetime.date:
    TODAY = datetime.datetime.now().date()
    BEGIN_SEARCH = TODAY - datetime.timedelta(days=14)
    start_date = BEGIN_SEARCH
    end_date = TODAY

    # I assume there is a better way to figure this out?
    # what is the most recent price we have for apple
    feature_value_map = (
        get_FeaturesDAO()
        .get_feature_data(
            gbi_ids=[714],
            features=["spiq_close"],
            start_date=start_date,
            end_date=end_date,
        )
        .get()
    )

    latest_datetime = feature_value_map["spiq_close"].index[-1].to_pydatetime()
    return latest_datetime.date()


async def get_latest_price(context: PlanRunContext, stock_ids: List[StockID]) -> Dict[int, float]:
    statistic_id = StatisticId(stat_id="spiq_close", stat_name="Price")
    TODAY = datetime.date.today()
    start_date = TODAY
    end_date = TODAY
    data_table = await get_statistic_data(
        context=context,
        statistic_id=statistic_id,
        start_date=start_date,
        end_date=end_date,
        stock_ids=stock_ids,
        ffill_days=7,  # to cover long holidays, check 7 days back for the last updated price
    )

    # Extract relevant columns
    stock_column: List[StockID] = data_table.get_stocks()
    price_column: List[float] = data_table.columns[2].data  # type: ignore

    # Store the latest price for each stock
    price_map = {}

    for stock, price in zip(stock_column, price_column):
        stock_gbi_id = stock.gbi_id
        price_map[stock_gbi_id] = price

    return price_map
