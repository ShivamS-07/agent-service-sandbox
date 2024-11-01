import datetime as dt
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import backoff
import numpy as np
from feature_service_proto_v1.feature_service_pb2 import GetFeatureDataResponse
from gbi_common_py_utils.numpy_common import NumpyCube

from agent_service.external.feature_svc_client import (
    get_feature_data,
    nc_swap_rows_fields,
)
from agent_service.utils.async_utils import async_wrap
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.feature_flags import get_ld_flag

logger = logging.getLogger(__name__)
EARLIEST_START_DATE = dt.date(1990, 1, 1)
_dao = None

# need to expose this type to the type checker but avoid importing it until needed normally
if TYPE_CHECKING:
    from data_access_layer.core.dao.features.stock_search_dao import StockSearchDAO


@backoff.on_exception(
    backoff.expo,
    exception=Exception,
    max_time=int(dt.timedelta(seconds=120).total_seconds()),
    max_tries=10,
    logger=logger,
)
def get_stock_search_dao() -> "StockSearchDAO":
    from data_access_layer.core.dao.features.stock_search_dao import StockSearchDAO

    global _dao
    if not _dao:
        _dao = StockSearchDAO()
    return _dao


@async_wrap
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
    end_date = end_date or get_now_utc().date()
    records = get_stock_search_dao().get_feature_data(
        gbi_ids=gbi_ids, features=features, start_date=start_date, end_date=end_date
    )
    return records


async def get_feature_svc_stock_volume(user_id: str, gbi_ids: List[int]) -> List[Dict[str, Any]]:
    # mimic the behavior of stock search version of this function
    if not gbi_ids:
        return []

    # calendar days (prob not equivalent to stocksearch value but serves the same purpose)
    num_days = 63
    end_date = get_now_utc().date()
    start_date = end_date - dt.timedelta(days=num_days)
    output_currency = "USD"
    features = ["spiq_real_volume", "spiq_close"]
    use_natural_axis = False  # Daily

    ffill_days_val = num_days

    resp: GetFeatureDataResponse = await get_feature_data(
        user_id=user_id,
        statistic_ids=features,
        stock_ids=gbi_ids,
        from_date=start_date,
        to_date=end_date,
        ffill_days=ffill_days_val,
        use_natural_axis=use_natural_axis,
        target_currency=output_currency,
    )

    security_data = resp.security_data[0].data_cube
    nc = NumpyCube.initialize_from_proto_bytes(data=security_data, cols_are_dates=True)  # type: ignore # noqa
    nc_swap_rows_fields(nc)

    result = []

    # calc the average dollar volume for each stock
    for gbi_id in nc.rows:
        dol_vol = (
            nc.np_data[nc.row_map[gbi_id], :, nc.field_map["spiq_close"]]
            * nc.np_data[nc.row_map[gbi_id], :, nc.field_map["spiq_real_volume"]]
        )
        avg_dol_vol = np.nanmean(dol_vol)

        # the nupycube comes across the wire with  string gbi_ids
        d = {"gbi_id": int(gbi_id), "volume_63_day": avg_dol_vol}
        result.append(d)

    return result


async def sort_stocks_by_volume(user_id: str, gbi_ids: List[int]) -> List[Tuple[int, float]]:
    if get_ld_flag(
        flag_name="use-aurora-stock-volume-info",
        default=False,
        user_context=None,  # wont be releasing this to individuals
    ):
        a_data = await get_feature_svc_stock_volume(user_id, gbi_ids=gbi_ids)
        ffill2 = {
            d["gbi_id"]: d["volume_63_day"]
            for d in a_data
            if d["volume_63_day"] is not None and not np.isnan(d["volume_63_day"])
        }
        sorted_list2 = sorted(ffill2.items(), key=lambda x: x[1], reverse=True)
        return sorted_list2

    END_DATE = get_now_utc().date()
    # look back N days to forwardfill
    START_DATE = END_DATE - dt.timedelta(days=21)

    data = await get_stock_search_data(
        gbi_ids=gbi_ids,
        features=["volume_63_day"],
        start_date=START_DATE,
        end_date=END_DATE,
    )

    ffill = {d["gbi_id"]: d["volume_63_day"] for d in data if d["volume_63_day"] is not None}
    sorted_list = sorted(ffill.items(), key=lambda x: x[1], reverse=True)
    return sorted_list


async def async_sort_stocks_by_volume(user_id: str, gbi_ids: List[int]) -> List[Tuple[int, float]]:
    # stock search can potentially be slow so wrap this in async
    return await sort_stocks_by_volume(user_id, gbi_ids)
