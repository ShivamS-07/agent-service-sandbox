import datetime
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional

from agent_service.utils.clickhouse import Clickhouse, VisAlphaDataset
from agent_service.utils.constants import UNITS_LOOKUP, VA_NO_CURRENCY_PLACEHOLDER
from agent_service.utils.date_utils import get_year_quarter_for_date

logger = logging.getLogger(__name__)


class VisibleAlphaVID(IntEnum):
    KEY_VALUE = 9999999


@dataclass
class KPIMetadata:
    name: str
    pid: int
    dataset: VisAlphaDataset


@dataclass
class KPIDatapoint:
    pid: int
    quarter: int
    year: int
    value: float
    unit: str
    name: Optional[str] = None
    long_unit: Optional[str] = None
    currency: Optional[str] = None
    is_estimate: bool = False


def _generate_periods(
    starting_quarter: int,
    starting_year: int,
    num_prev_quarters: int,
    num_future_quarters: int,
    use_fiscal_periods: bool = False,
) -> List[str]:
    periods = []
    quarter = starting_quarter
    year = starting_year
    for _ in range(num_prev_quarters + 1):
        if use_fiscal_periods:
            period = f"{quarter}QFY-{year}"
        else:
            # Use the calendar period instead
            period = f"{quarter}QCY-{year}"
        periods.append(period)
        if quarter == 1:
            quarter = 4
            year -= 1
        else:
            quarter -= 1

    # Reverse the list so it's oldest to newest and drop the starting quarter-year
    # since it'll get readded at the start of the subsequent loop
    periods = periods[::-1][:-1]
    quarter = starting_quarter
    year = starting_year
    for _ in range(num_future_quarters + 1):
        if use_fiscal_periods:
            period = f"{quarter}QFY-{year}"
        else:
            # Use the calendar period instead
            period = f"{quarter}QCY-{year}"
        periods.append(period)
        if quarter == 1:
            quarter = 4
            year += 1
        else:
            quarter += 1

    # Reversing the list so it's in oldest to earliest
    return periods


class VisAlphaClickHouseClient:
    def __init__(self) -> None:
        self.va_ch = Clickhouse(environment="DEV")
        self.gbi_to_cid_mapping: Dict[int, str] = {}

    def _kpi_json_to_dataclass(self, kpi_json: Dict[str, Any], is_estimate: bool) -> KPIDatapoint:
        unit_id = int(kpi_json["UnitId"])
        currency = kpi_json.get("Currency", "")
        name = kpi_json.get("ParameterName", "")
        if currency == VA_NO_CURRENCY_PLACEHOLDER:
            currency = ""
        return KPIDatapoint(
            pid=kpi_json["ParameterId"],
            value=float(kpi_json["Value"]),
            quarter=int(kpi_json["Period"][0]),
            year=int(kpi_json["Period"][-4:]),
            is_estimate=is_estimate,
            unit=UNITS_LOOKUP[unit_id].shorthand,
            currency=currency,
            long_unit=UNITS_LOOKUP[unit_id].name,
            name=name,
        )

    async def get_cid_from_gbi_id(self, gbi_ids: List[int]) -> Dict[int, str]:
        # TODO: Neet to handle gbis when running on prod, currently assumes gbis are from dev
        gbi_cid_lookup = await self.va_ch.get_cid_for_gbi_ids(gbi_ids)
        if not gbi_cid_lookup:
            return {}
        return gbi_cid_lookup

    async def get_company_kpis_for_gbi(
        self,
        gbi_id: int,
        vid_filter: Optional[VisibleAlphaVID] = None,
        pids_filter: Optional[List[int]] = None,
    ) -> List[KPIMetadata]:
        # TODO: make this work for multiple gbi_ids
        cid = (await self.get_cid_from_gbi_id([gbi_id])).get(gbi_id)
        if cid is None:
            return []

        if (vid_filter is not None) and (pids_filter is not None):
            raise ValueError("Cannot have both a vid filter and a pid filter at the same time!")

        if vid_filter is not None:
            line_items = await self.va_ch.get_company_data_kpis(cid, vid=str(vid_filter.value))
        elif pids_filter is not None:
            line_items = await self.va_ch.get_company_data_kpis(
                cid, pids=list(map(str, pids_filter))
            )
        else:
            line_items = await self.va_ch.get_company_data_kpis(cid)
        return [
            KPIMetadata(
                line_item["ParameterName"],
                int(line_item["ParameterId"]),
                VisAlphaDataset.COMPANY_DATASET,
            )
            for line_item in line_items
        ]

    async def get_company_kpis_for_gbis(
        self,
        gbi_ids: List[int],
    ) -> Dict[int, List[KPIMetadata]]:
        # TODO: make this work for multiple gbi_ids
        gbi_cid_lookup = await self.get_cid_from_gbi_id(gbi_ids)
        cid_gbi_lookup = {cid: gbi for gbi, cid in gbi_cid_lookup.items()}
        cids = list(gbi_cid_lookup.values())

        unsorted_pids_list = await self.va_ch.get_company_data_kpis_for_multiple_cids(cids)

        gbi_kpi_lookup: Dict[int, List[KPIMetadata]] = {gbi: [] for gbi in gbi_ids}
        for pid_data in unsorted_pids_list:
            gbi_id = cid_gbi_lookup[pid_data["VACompanyId"]]
            gbi_kpi_lookup[gbi_id].append(
                KPIMetadata(
                    pid_data["ParameterName"],
                    int(pid_data["ParameterId"]),
                    VisAlphaDataset.COMPANY_DATASET,
                )
            )
        return gbi_kpi_lookup

    async def fetch_data_for_kpis(
        self,
        gbi_id: int,
        kpis: List[KPIMetadata],
        num_prev_quarters: int = 0,
        num_future_quarters: int = 0,
        starting_date: Optional[datetime.date] = None,
        quarter: Optional[int] = None,
        year: Optional[int] = None,
        dataset: VisAlphaDataset = VisAlphaDataset.COMPANY_DATASET,
        estimate: bool = False,
    ) -> Dict[int, List[KPIDatapoint]]:
        """
        For a given stock and either a maximum date or a year quarter, return
        prior KPI's sorted in chronological order.
        """
        if not quarter and not year:
            assert (
                starting_date is not None
            ), "Must pass in starting_date if no year or quarter specified"

        if not starting_date:
            assert (
                year is not None
            ), "Must pass in year and quarter if starting_date is not specified"
            assert (
                quarter is not None
            ), "Must pass in year and quarter if starting_date is not specified"

        # TODO: Neet to handle gbis when running on prod, currently assumes gbis are from dev
        gbi_cid_lookup = await self.va_ch.get_cid_for_gbi_ids([gbi_id])
        if not gbi_cid_lookup:
            return {}
        cid = gbi_cid_lookup[gbi_id]
        pids = [str(kpi.pid) for kpi in kpis]
        periods = []
        fiscal_year = False
        if quarter and year:
            fiscal_year = True
            periods = _generate_periods(
                quarter, year, num_prev_quarters, num_future_quarters, use_fiscal_periods=True
            )
        elif starting_date:
            fiscal_year = False
            year, quarter = get_year_quarter_for_date(starting_date)
            periods = _generate_periods(quarter, year, num_prev_quarters, num_future_quarters)

        kpi_datapoints = defaultdict(list)
        data = await self.va_ch.get_kpi_values_for_cid(
            cid=cid,
            pids=pids,
            periods=periods,
            dataset=dataset,
            estimate=estimate,
            use_fiscal_year=fiscal_year,
        )
        for kpi_json in data:
            if kpi_json.get("Value") is not None:
                parameter_id = int(kpi_json["ParameterId"])
                kpi_datapoints[parameter_id].append(
                    self._kpi_json_to_dataclass(kpi_json=kpi_json, is_estimate=estimate)
                )
            else:
                logger.warning(
                    f"{gbi_id} has no {'estimate' if estimate else 'actual'} "
                    f"value for {kpi_json['ParameterId']} for {kpi_json['Period']}"
                )

        for parameter_id, kpi_instances in kpi_datapoints.items():
            kpi_datapoints[parameter_id] = sorted(
                kpi_instances, key=lambda data: (data.year, data.quarter)
            )
        return kpi_datapoints

    async def fetch_data_for_company_kpis_bulk(
        self,
        gbi_pids_dict: Dict[int, List[KPIMetadata]],
        starting_date: datetime.date,
        num_prev_quarters: int = 0,
        num_future_quarters: int = 0,
        estimate: bool = False,
    ) -> Dict[int, List[KPIDatapoint]]:
        """
        Fetch KPI's for a set of stocks. Note that all kpi's specified MUST be
        in the visible alpha standard dataset so that they are common to all
        stocks.
        Returns a map from GBI ID to a list of datapoints sorted in
        chronological order.
        """
        gbi_ids = list(gbi_pids_dict.keys())
        year, quarter = get_year_quarter_for_date(starting_date)
        gbi_to_cid_map = await self.va_ch.get_cid_for_gbi_ids(gbi_ids)
        cid_to_gbi_map = {cid: gbi for gbi, cid in gbi_to_cid_map.items()}
        cid_pids_dict = {
            gbi_to_cid_map[gbi]: [pid.pid for pid in pids]
            for gbi, pids in gbi_pids_dict.items()
            if gbi in gbi_to_cid_map
        }
        periods = _generate_periods(
            quarter, year, num_prev_quarters, num_future_quarters, use_fiscal_periods=False
        )
        data = await self.va_ch.get_company_kpi_values_for_cids(
            cid_pids_dict=cid_pids_dict, periods=periods, estimate=estimate, use_fiscal_year=False
        )
        kpi_datapoints: Dict[int, List[KPIDatapoint]] = defaultdict(list)
        for kpi_json in data:
            if kpi_json.get("Value") is not None:
                gbi = cid_to_gbi_map[kpi_json["VACompanyId"]]
                datapoint = self._kpi_json_to_dataclass(kpi_json, estimate)
                kpi_datapoints[gbi].append(datapoint)
        for gbi_id, datapoints in kpi_datapoints.items():
            kpi_datapoints[gbi_id] = sorted(datapoints, key=lambda data: (data.year, data.quarter))
        return kpi_datapoints


if __name__ == "__main__":
    va = VisAlphaClickHouseClient()
    data = va.get_company_kpis_for_gbis(gbi_ids=[714, 7555])
