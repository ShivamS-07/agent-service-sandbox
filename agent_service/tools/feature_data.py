import datetime
from threading import Lock
from typing import List, Optional

import pandas as pd
from cachetools import TTLCache, cached
from data_access_layer.core.dao.features.feature_utils import get_feature_metadata
from data_access_layer.core.dao.features.features_dao import FeaturesDAO

from agent_service.io_types.table import Table, TableColumn, TableColumnType
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.stocks import StatisticsIdentifierLookupInput
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import async_wrap
from agent_service.utils.postgres import get_psql

LATEST_DATE_SECONDS = 60 * 60
TTLCache(maxsize=1, ttl=LATEST_DATE_SECONDS)


# instantiate globally so that metadata caches do not have to be created on each
# new request
FEATURES_DAO = FeaturesDAO()


@tool(
    description=(
        "This function takes a string (close price, Churn low, Market Capitalization, Coppock Curve, e.g.)"
        " which refers to a financial indicator or statistic, and matches it to an identifier for that "
        "statistic. You should use this before fetching any data for the statistic."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def statistic_identifier_lookup(
    args: StatisticsIdentifierLookupInput, context: PlanRunContext
) -> str:
    """Returns the string identifier of a statistic given its name (Churn low, Market Capitalization, e.g.)

    This function performs word similarity name match to find the statistic's identifier.


    Args:
        args (StatisticsIdentifierLookupInput): The input arguments for the statistic lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        str: The integer identifier of the statistic.
    """
    db = get_psql()
    # TODO :
    # 1. Add more filtering or new column (agent_supported) to table
    # 2. implement solar solution

    # Word similarity name match
    sql = """
    SELECT id FROM public.features feat
    WHERE feat.data_provider = 'SPIQ'
    ORDER BY word_similarity(lower(feat.name), lower(%s)) DESC
    LIMIT 1
    """
    rows = db.generic_read(sql, [args.statistic_name])
    if rows:
        return rows[0]["id"]

    raise ValueError(f"Could not find the stock {args.statistic_name}")


class FeatureDataInput(ToolArgs):
    stock_ids: List[int]
    field_id: str
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    # in the future we may want to take a currency as well


@tool(
    description=(
        "This function returns the time series of data for a statistic_id"
        " for each stock in the list of stocks_ids."
        " Optionally a start_date and end_date may be provided to specify a date range"
        " to get a specific date only set both inputs to the same date."
        " If neither date is provided then it will assume the request "
        "is for the most recent date for which data exists. The statistic_id MUST be "
        "fetched with the lookup function, it cannot be an arbitrary string."
    ),
    category=ToolCategory.STATISTICS,
    tool_registry=ToolRegistry,
)
async def get_statistic_data_for_companies(
    args: FeatureDataInput, context: PlanRunContext
) -> Table:
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
    return await _async_get_feature_data(args, context)


@async_wrap
def _async_get_feature_data(args: FeatureDataInput, context: PlanRunContext) -> Table:
    # if getting a large number of features or stocks or dates, feature dao could be slow-ish
    # so  provide this wrapper to not block the event loop
    return _sync_get_feature_data(args, context)


def _sync_get_feature_data(args: FeatureDataInput, context: PlanRunContext) -> Table:

    if args.end_date is None and args.start_date is not None:
        args.end_date = datetime.date.today()

    features_metadata = get_feature_metadata(feature_ids=[args.field_id])
    metadata = features_metadata.get(args.field_id, None)
    # print("--")
    # print("metadata", metadata)
    source = metadata.source if metadata else "NO_SUCH_SOURCE"
    supported_sources = ["SPIQ_DAILY", "SPIQ_DIVIDEND", "SPIQ_TARGET", "SPIQ_QUARTERLY"]
    if source not in supported_sources:
        raise ValueError(f"Data field: {args.field_id} is from an unsupported source: {source}")

    if source == "SPIQ_DAILY":
        df = get_daily_feature_data(args, context)

    elif source in ["SPIQ_DIVIDEND", "SPIQ_TARGET"]:
        df = get_non_daily_data(args, context)

    elif source == "SPIQ_QUARTERLY":
        df = get_quarterly_data(args, context)

    else:
        raise ValueError(
            f"code path missing: Data field: {args.field_id} is from an unsupported source: {source}"
        )

    df.index.rename("Date", inplace=True)
    return Table(
        data=df,
        columns=[TableColumn(label="Date", col_type=TableColumnType.DATE, is_indexed=True)]
        + [
            # TODO handle smarter column types, etc.
            TableColumn(
                label=col if not isinstance(col, tuple) else col[1],
                col_label_is_stock_id=True,
                col_type=TableColumnType.FLOAT,
            )
            for col in df.columns
        ],
    )


def get_daily_feature_data(args: FeatureDataInput, context: PlanRunContext) -> pd.DataFrame:
    # if no dates are given assume they just want the latest value
    LATEST_DATE = get_latest_date()
    start_date = LATEST_DATE
    end_date = LATEST_DATE

    # if only 1 date is given use that as start & end
    if args.start_date is None and args.end_date is not None:
        start_date = args.end_date
        end_date = start_date
    elif args.start_date is not None and args.end_date is None:
        start_date = args.start_date
        end_date = start_date
    elif args.start_date is not None and args.end_date is not None:
        # if both dates are given use as is
        start_date = args.start_date
        end_date = args.end_date

    # TODO: validate start <= end
    # dates are not in the future, etc

    FORWARD_FILL_LOOKBACK = datetime.timedelta(days=14)
    lookup_start_date = start_date - FORWARD_FILL_LOOKBACK
    feature_value_map = FEATURES_DAO.get_feature_data(
        gbi_ids=args.stock_ids,
        features=[args.field_id],
        start_date=lookup_start_date,
        end_date=end_date,
        # currency=output_currency,
    ).get()

    raw_df = feature_value_map[args.field_id]
    # print("raw_df", raw_df)
    idx_daily = pd.date_range(start=lookup_start_date, end=end_date, freq="D")
    df = raw_df.reindex(idx_daily, method="ffill")
    df.index.names = ["date"]

    # cut it back down to the requested time range
    df = df.loc[pd.to_datetime(start_date) : pd.to_datetime(end_date)]
    return df


def get_non_daily_data(args: FeatureDataInput, context: PlanRunContext) -> pd.DataFrame:
    # if no dates are given assume they just want the latest value
    LATEST_DATE = get_latest_date()
    start_date = LATEST_DATE
    end_date = LATEST_DATE

    # if only 1 date is given use that as start & end
    if args.start_date is None and args.end_date is not None:
        start_date = args.end_date
        end_date = start_date
    elif args.start_date is not None and args.end_date is None:
        start_date = args.start_date
        end_date = start_date
    elif args.start_date is not None and args.end_date is not None:
        # if both dates are given use as is
        start_date = args.start_date
        end_date = args.end_date

    # TODO: validate start <= end
    # dates are not in the future, etc

    FORWARD_FILL_LOOKBACK = datetime.timedelta(days=366)
    lookup_start_date = start_date - FORWARD_FILL_LOOKBACK
    feature_value_map = FEATURES_DAO.get_feature_data(
        gbi_ids=args.stock_ids,
        features=[args.field_id],
        start_date=lookup_start_date,
        end_date=end_date,
        # currency=output_currency,
    ).get()

    raw_df = feature_value_map[args.field_id]
    # print("raw_df", raw_df)

    df = raw_df

    if start_date == end_date:
        # get the latest
        df = df.iloc[[-1]]
    else:
        # get only the range they asked for
        df = df.loc[pd.to_datetime(start_date) : pd.to_datetime(end_date)]

    return df


# this might need a separate tool for addressing the data via relative and absolute periods
def get_quarterly_data(args: FeatureDataInput, context: PlanRunContext) -> pd.DataFrame:
    # if no dates are given assume they just want the latest value
    LATEST_DATE = get_latest_date()
    start_date = LATEST_DATE
    end_date = LATEST_DATE

    # if only 1 date is given use that as start & end
    if args.start_date is None and args.end_date is not None:
        start_date = args.end_date
        end_date = start_date
    elif args.start_date is not None and args.end_date is None:
        start_date = args.start_date
        end_date = start_date
    elif args.start_date is not None and args.end_date is not None:
        # if both dates are given use as is
        start_date = args.start_date
        end_date = args.end_date

    if start_date != end_date:
        raise ValueError(
            "Quarterly Data is currently only available for single points"
            f" in time field: {args.field_id}"
        )
    # TODO: validate start <= end
    # dates are not in the future, etc

    LOOKBACK = datetime.timedelta(days=366)
    lookup_start_date = start_date - LOOKBACK
    feature_value_map = FEATURES_DAO.get_feature_data(
        gbi_ids=args.stock_ids,
        features=[args.field_id],
        start_date=lookup_start_date,
        end_date=end_date,
        # currency=output_currency,
    ).get()

    raw_df = feature_value_map[args.field_id]
    # print("raw_df", raw_df)

    df = raw_df

    if start_date == end_date:
        # get the latest
        df = df.iloc[[-1]]
    else:
        # this is not yet supported
        # we will need new DAL apis to get the data indexed by calendar quarter
        # and pick the latest value for each one
        pass

    rel_per = -1
    idx = pd.IndexSlice
    # grab all the rows "idx[:] ..."
    # grab only the -1 relperiod column "... idx[rel_per:-1, ..."
    # grab all the gbiids "... :]"
    df = df.loc[idx[:], idx[rel_per:-1, :]]  # type: ignore
    return df


# TODO in the future we need a latest date per region
@cached(cache=TTLCache(maxsize=1, ttl=LATEST_DATE_SECONDS), lock=Lock())
def get_latest_date() -> datetime.date:
    TODAY = datetime.datetime.now().date()
    BEGIN_SEARCH = TODAY - datetime.timedelta(days=14)
    start_date = BEGIN_SEARCH
    end_date = TODAY

    # I assume there is a better way to figure this out?
    # what is the most recent price we have for apple
    feature_value_map = FEATURES_DAO.get_feature_data(
        gbi_ids=[714],
        features=["spiq_close"],
        start_date=start_date,
        end_date=end_date,
    ).get()

    latest_datetime = feature_value_map["spiq_close"].index[-1].to_pydatetime()
    return latest_datetime.date()
