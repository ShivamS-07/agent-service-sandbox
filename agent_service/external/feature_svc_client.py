import datetime
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, Generator, List, Optional, Set, Tuple

import pandas as pd
from feature_service_proto_v1.daily_pricing_pb2 import (
    CubeFormat,
    GetAdjustedCumulativeReturnsRequest,
    GetAdjustedCumulativeReturnsResponse,
)
from feature_service_proto_v1.earnings_pb2 import (
    GetEarningsReleasesInRangeRequest,
    GetEarningsReleasesInRangeResponse,
)
from feature_service_proto_v1.feature_metadata_service_grpc import (
    FeatureMetadataServiceStub,
)
from feature_service_proto_v1.feature_metadata_service_pb2 import (
    GetAllFeaturesMetadataRequest,
    GetAllFeaturesMetadataResponse,
    GetFeatureHierarchyRequest,
    GetFeatureHierarchyResponse,
)
from feature_service_proto_v1.feature_service_grpc import FeatureServiceStub
from feature_service_proto_v1.feature_service_pb2 import (
    GetFeatureDataRequest,
    GetFeatureDataResponse,
    TimeAxis,
)
from feature_service_proto_v1.proto_cube_pb2 import ProtoCube
from feature_service_proto_v1.real_time_pricing_pb2 import (
    GetRealTimePricesRequest,
    GetRealTimePricesResponse,
)
from gbi_common_py_utils.numpy_common import NumpyCube
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    get_environment_tag,
)
from google.protobuf.json_format import MessageToDict
from grpclib.client import Channel

from agent_service.external.grpc_utils import (
    date_to_timestamp,
    get_default_grpc_metadata,
    grpc_retry,
    timestamp_to_date,
)
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)

DEFAULT_URLS = {
    LOCAL_TAG: ("feature-service-dev.boosted.ai", 50051),
    DEV_TAG: ("feature-service-dev.boosted.ai", 50051),
    PROD_TAG: ("feature-service-prod.boosted.ai", 50051),
}

METADATA_DEFAULT_URLS = {
    LOCAL_TAG: ("feature-metadata-service-dev.boosted.ai", 50051),
    DEV_TAG: ("feature-metadata-service-dev.boosted.ai", 50051),
    PROD_TAG: ("feature-metadata-service-prod.boosted.ai", 50051),
}


@lru_cache(maxsize=1)
def get_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("FEATURE_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found FEATURE_SERVICE_URL override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return DEFAULT_URLS[env]


@lru_cache(maxsize=1)
def get_metadata_url_and_port() -> Tuple[str, int]:
    url = os.environ.get("FEATURE_METADATA_SERVICE_URL")
    if url is not None:
        logger.warning(f"Found FEATURE_METADATA_SERVICE_URL override: {url}")
        split_url, port = url.split(":")
        return split_url, int(port)

    env = get_environment_tag()
    if env not in METADATA_DEFAULT_URLS:
        raise ValueError(f"No URL is set for environment {env}")
    return METADATA_DEFAULT_URLS[env]


@contextmanager
def _get_service_stub() -> Generator[FeatureServiceStub, None, None]:
    try:
        url, port = get_url_and_port()
        channel = Channel(url, port)
        yield FeatureServiceStub(channel)
    finally:
        channel.close()


@contextmanager
def _get_metadata_service_stub() -> Generator[FeatureMetadataServiceStub, None, None]:
    try:
        url, port = get_metadata_url_and_port()
        channel = Channel(url, port)
        yield FeatureMetadataServiceStub(channel)
    finally:
        channel.close()


@grpc_retry
@async_perf_logger
async def get_feature_data(
    user_id: str,
    statistic_ids: List[str],
    stock_ids: List[int],
    from_date: datetime.date,
    to_date: datetime.date,
    target_currency: Optional[str] = None,
    ffill_days: int = 0,
    use_natural_axis: bool = False,
) -> GetFeatureDataResponse:
    """
    Optional Params Description:

    target_currency: currency to convert all outputs to. Otherwise outputs are
        converted to the currency of the first stock in the list.
    ffill_days: number of days to forward fill missing data.
    use_natural_axis: if True, feature service will return axes in quarters (YYYYQ)
        or Fiscal Quarters (FYYYYQ) instead of dates as axes depending on the best
        display format of the specific statistic (feature).
    """
    with _get_service_stub() as stub:
        req = GetFeatureDataRequest(
            feature_ids=statistic_ids,
            gbi_ids=stock_ids,
            from_time=date_to_timestamp(from_date),
            to_time=date_to_timestamp(to_date),
            iso_currency=target_currency,
            forward_fill_days=ffill_days,
            prefer_time_axis=(
                TimeAxis.TIME_AXIS_BEST_HUMAN_READABLE
                if use_natural_axis
                else TimeAxis.TIME_AXIS_DATE
            ),
        )
        resp: GetFeatureDataResponse = await stub.GetFeatureData(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def get_all_variables_metadata(user_id: str) -> GetAllFeaturesMetadataResponse:
    with _get_metadata_service_stub() as stub:
        req = GetAllFeaturesMetadataRequest(filter_agent_enabled=True)
        resp: GetAllFeaturesMetadataResponse = await stub.GetAllFeaturesMetadata(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def get_all_variable_hierarchy(user_id: str) -> GetFeatureHierarchyResponse:
    with _get_metadata_service_stub() as stub:
        req = GetFeatureHierarchyRequest()
        resp: GetFeatureHierarchyResponse = await stub.GetFeatureHierarchy(
            req,
            metadata=get_default_grpc_metadata(user_id=user_id),
        )
        return resp


@grpc_retry
@async_perf_logger
async def get_earnings_releases_in_range(
    gbi_ids: List[int], start_date: datetime.date, end_date: datetime.date, user_id: str = ""
) -> Dict[int, Set[Tuple[datetime.date, Optional[datetime.datetime]]]]:
    """
    Given a list of stocks and a date range, return a mapping from gbi id to a
    list of earnings release dates within the range. Note that if no earnings
    call for a stock appears in the range, it will not be in the output dict.
    """
    with _get_service_stub() as stub:
        req = GetEarningsReleasesInRangeRequest(
            gbi_ids=gbi_ids,
            start_date=date_to_timestamp(start_date),
            end_date=date_to_timestamp(end_date),
        )
        resp: GetEarningsReleasesInRangeResponse = await stub.GetEarningsReleasesInRange(
            req, metadata=get_default_grpc_metadata(user_id=user_id)
        )
    output: Dict[int, Set[Tuple[datetime.date, Optional[datetime.datetime]]]] = defaultdict(set)
    for date_releases in resp.data:
        earnings_time_lookup: Dict[int, datetime.datetime] = {}
        date = timestamp_to_date(date_releases.release_date)  # This will only have YY MM DD
        if date:
            for timestamp in date_releases.earnings_timestamps:
                if timestamp.timestamp:
                    converted_datetime = timestamp.timestamp.ToDatetime()
                    if converted_datetime != datetime.datetime(1970, 1, 1, 0, 0):
                        earnings_time_lookup[timestamp.gbi_id] = converted_datetime

            for gbi_id in date_releases.gbi_ids:
                earnings_datetime = earnings_time_lookup.get(gbi_id, None)
                if earnings_datetime:
                    output[gbi_id].add((date, earnings_datetime))
                else:
                    output[gbi_id].add((date, None))

    return output


@grpc_retry
@async_perf_logger
async def get_return_for_stocks(
    gbi_ids: List[int], start_date: datetime.date, end_date: datetime.date, user_id: str = ""
) -> Dict[int, float]:
    """
    Given a list of stocks and a date range, returns the adjusted cumulative returns
    for the stocks in the range. The output is a DataFrame with the following columns:
    - gbi_id: the stock id
    - Date: the most recent date in the range
    - Return: the adjusted cumulative return for the stock on the given date

    """

    def proto_cube_to_dataframe(proto_cube: ProtoCube) -> Dict[int, float]:
        # Convert ProtoCube to dictionary
        proto_dict = MessageToDict(proto_cube)

        # Extract the necessary data
        rows = proto_dict.get("rows", [])
        columns = proto_dict.get("columns", [])
        fields = proto_dict.get("fields", [])
        data = proto_dict.get("data", [])

        # Prepare a list to gather data
        records = []

        # Populate the list with the necessary data
        for i, row in enumerate(data):
            for j, col in enumerate(row["columns"]):
                field_values = col["fields"]
                for k, field_value in enumerate(field_values):
                    records.append({"Row": rows[i], "Date": columns[j], fields[k]: field_value})

        # Create a DataFrame from the list
        df_long = pd.DataFrame(records)
        # Convert 'Date' column to datetime
        df_long["Date"] = pd.to_datetime(df_long["Date"])
        df_long["Row"] = df_long["Row"].astype(int)
        df_long["adjusted_cumulative_return"] = df_long["adjusted_cumulative_return"].astype(float)
        # filter out rows with NaN values, reset index, rename columns, and drop 'Date' column
        df_res = (
            df_long.loc[df_long["adjusted_cumulative_return"].notna(), :]
            .loc[df_long["Date"] == df_long.Date.max(), :]
            .sort_values(by="adjusted_cumulative_return", ascending=False)
            .reset_index(drop=True)
            .rename(columns={"Row": "gbi_id", "adjusted_cumulative_return": "cum_return"})
            .drop(columns=["Date", "close_price", "dividend_amount"])
        )

        # convert df to dict mapping gbi_id to return
        dict_res = df_res.set_index("gbi_id")["cum_return"].to_dict()
        return dict_res

    with _get_service_stub() as stub:
        req = GetAdjustedCumulativeReturnsRequest(
            gbi_ids=gbi_ids,
            start_date=date_to_timestamp(start_date),
            end_date=date_to_timestamp(end_date),
            response_format=CubeFormat.CUBE_FORMAT_PROTOBUF,
        )
        resp: GetAdjustedCumulativeReturnsResponse = await stub.GetAdjustedCumulativeReturns(
            req, metadata=get_default_grpc_metadata(user_id=user_id)
        )
        if resp.status.code != 0:
            raise ValueError(
                f"Failed to get stock return: {resp.status.code} - {resp.status.message}"
            )
    return proto_cube_to_dataframe(resp.data)


def proto_cube_to_dataframe(proto_cube: ProtoCube) -> pd.DataFrame:
    # Convert ProtoCube to dictionary
    proto_dict = MessageToDict(proto_cube)

    # Extract the necessary data
    rows = proto_dict.get("rows", [])
    columns = proto_dict.get("columns", [])
    fields = proto_dict.get("fields", [])
    data = proto_dict.get("data", [])

    # Prepare a list to gather data
    records = []

    # Populate the list with the necessary data
    for i, row in enumerate(data):
        for j, col in enumerate(row["columns"]):
            field_values = col["fields"]
            for k, field_value in enumerate(field_values):
                records.append({"Row": rows[i], "Date": columns[j], fields[k]: field_value})

    # Create a DataFrame from the list
    df_long = pd.DataFrame(records)
    # Convert 'Date' column to datetime
    df_long["Date"] = pd.to_datetime(df_long["Date"])
    # filter out rows with NaN values, reset index, rename columns, and drop 'Date' column
    df_res = (
        df_long.loc[df_long["adjusted_cumulative_return"].notna(), :]
        .sort_values(by="Date", ascending=True)
        .reset_index(drop=True)
        .rename(columns={"Date": "date", "adjusted_cumulative_return": "cum_return"})
        .drop(columns=["Row"])
    )

    return df_res


@grpc_retry
@async_perf_logger
async def get_return_for_single_stock(
    gbi_id: int, start_date: datetime.date, end_date: datetime.date, user_id: str = ""
) -> pd.DataFrame:
    """
    Given a stock id (or universe id) and a date range, returns the adjusted cumulative returns
    for the stock in the date range. The output is a DataFrame with the following columns:
    - date: the most recent date in the range
    - cum_return: the adjusted cumulative return for the stock on the given date
    - dividend_amount: the dividend amount for the stock on the given date
    - close_price: the close price for the stock on the given date

    """
    with _get_service_stub() as stub:
        req = GetAdjustedCumulativeReturnsRequest(
            gbi_ids=[gbi_id],
            start_date=date_to_timestamp(start_date),
            end_date=date_to_timestamp(end_date),
            response_format=CubeFormat.CUBE_FORMAT_PROTOBUF,
        )
        resp: GetAdjustedCumulativeReturnsResponse = await stub.GetAdjustedCumulativeReturns(
            req, metadata=get_default_grpc_metadata(user_id=user_id)
        )
        if resp.status.code != 0:
            raise ValueError(
                f"Failed to get stock return: {resp.status.code} - {resp.status.message}"
            )
    return proto_cube_to_dataframe(resp.data)


@grpc_retry
@async_perf_logger
async def get_intraday_prices(gbi_ids: List[int]) -> Dict[int, float]:
    """
    Given a stock ids, returns the latest real-time prices for those stocks.
    """
    with _get_service_stub() as stub:
        req = GetRealTimePricesRequest(gbi_ids=gbi_ids)
        resp: GetRealTimePricesResponse = await stub.GetRealTimePrices(req)
        if resp.status.code != 0:
            raise ValueError(
                f"Failed to get realtime price: {resp.status.code} - {resp.status.message},  {gbi_ids=}"
            )
    return {price.gbi_id: price.latest_price for price in resp.prices}


def nc_swap_rows_fields(nc: NumpyCube) -> None:
    # featuresvc returns nc fields x dates x gbiids
    # pa always uses gbi x dates x fields

    # swap the data
    nc.np_data = nc.np_data.transpose(2, 1, 0)

    # swap the indices
    nc.rows, nc.fields = nc.fields, nc.rows
    nc.row_mask, nc.field_mask = nc.field_mask, nc.row_mask

    # reset some other numpycube internals based on the above
    nc.map_rows_and_columns_and_fields()
