import datetime
from threading import Lock
from typing import List, Optional

from cachetools import TTLCache, cached
from data_access_layer.core.dao.features.features_dao import FeaturesDAO

from agent_service.io_types import StockTimeSeriesTable
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import async_wrap

LATEST_DATE_SECONDS = 60 * 60
TTLCache(maxsize=1, ttl=LATEST_DATE_SECONDS)


# instantiate globally so that metadata caches do not have to be created on each
# new request
FEATURES_DAO = FeaturesDAO()


class FeatureDataInput(ToolArgs):
    stock_ids: List[int]
    field_id: str
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    # in the future we may want to take a currency as well


@tool(
    description="""
    This function returns the time series of data associated with the data field_id
    and for each stock in the list of stocks_ids
    Optionally a start_date and end_date may be provided to specify a date range
    To get a specific date only set both inputs to the same date
    If neither date is provided then it will assume the request is for the most recent date for which data exists
    """,
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def get_feature_data(args: FeatureDataInput, context: PlanRunContext) -> StockTimeSeriesTable:
    """Returns the Time series of data for the requested field for each of the stocks_ids
    Optionally a start_date and end_date may be provided to specify a date range
    To get a specific date only set both inputs to the same date
    if only one of them is filled in or not None then it will be assumed to be a single date to be returned
    If neither date is provided then it will assume the request is for the most recent date for which data exists

    Args:
        args (FeatureDataInput): The input arguments for the feature data retrieval.
        context (PlanRunContext): The context of the plan run.

    Returns:
        StockTimeSeriesTable: The requested data.
    """
    return await _async_get_feature_data(args, context)


@async_wrap
def _async_get_feature_data(
    args: FeatureDataInput, context: PlanRunContext
) -> StockTimeSeriesTable:
    # if getting a large number of features or stocks or dates, feature dao could be slow-ish
    # so  provide this wrapper to not block the event loop
    return _sync_get_feature_data(args, context)


def _sync_get_feature_data(args: FeatureDataInput, context: PlanRunContext) -> StockTimeSeriesTable:
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

    feature_value_map = FEATURES_DAO.get_feature_data(
        gbi_ids=args.stock_ids,
        features=[args.field_id],
        start_date=start_date,
        end_date=end_date,
        # currency=output_currency,
    ).get()

    return StockTimeSeriesTable(val=feature_value_map[args.field_id])


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
