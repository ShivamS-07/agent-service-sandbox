import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Tuple

import backoff
from data_access_layer.core.dao.features.stock_search_dao import StockSearchDAO

from agent_service.utils.async_utils import async_wrap

logger = logging.getLogger(__name__)
EARLIEST_START_DATE = dt.date(1990, 1, 1)
dao = None


@backoff.on_exception(
    backoff.expo,
    exception=Exception,
    max_time=int(dt.timedelta(seconds=120).total_seconds()),
    max_tries=10,
    logger=logger,
)
def get_stock_search_dao() -> StockSearchDAO:
    global dao
    if not dao:
        dao = StockSearchDAO()
    return dao


@backoff.on_exception(
    backoff.expo,
    exception=Exception,
    max_time=int(dt.timedelta(seconds=120).total_seconds()),
    max_tries=10,
    logger=logger,
)
def get_stock_search_data(
    gbi_ids: List[int],
    features: List[str],
    start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None,
) -> List[Dict[str, Any]]:
    start_date = start_date or EARLIEST_START_DATE
    end_date = end_date or dt.datetime.now().date()
    records = get_stock_search_dao().get_feature_data(
        gbi_ids=gbi_ids, features=features, start_date=start_date, end_date=end_date
    )
    return records


def sort_stocks_by_volume(gbi_ids: List[int]) -> List[Tuple[int, float]]:
    END_DATE = dt.datetime.now().date()
    # look back N days to forwardfill
    START_DATE = END_DATE - dt.timedelta(days=21)

    data = get_stock_search_data(
        gbi_ids=gbi_ids,
        features=["volume_63_day"],
        start_date=START_DATE,
        end_date=END_DATE,
    )

    ffill = {d["gbi_id"]: d["volume_63_day"] for d in data if d["volume_63_day"] is not None}
    sorted_list = sorted(ffill.items(), key=lambda x: x[1], reverse=True)
    return sorted_list


@async_wrap
def async_sort_stocks_by_volume(gbi_ids: List[int]) -> List[Tuple[int, float]]:
    # stock search can potentially be slow so wrap this in async
    return sort_stocks_by_volume(gbi_ids)
