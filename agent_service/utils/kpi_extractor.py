import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from agent_service.io_types.text import KPIText
from agent_service.utils.clickhouse import VisAlphaDataset
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.visible_alpha import (
    KPIDatapoint,
    KPIMetadata,
    VisAlphaClickHouseClient,
)

logger = logging.getLogger(__name__)


@dataclass
class KPIInstance:
    name: str
    quarter: int
    year: int
    estimate: float
    actual: Optional[float] = None
    surprise: Optional[float] = None  # (Actual - Estimate) / |Estimate|
    unit: Optional[str] = None
    long_unit: Optional[str] = None
    currency: Optional[str] = None
    is_important_segment: bool = False  # Specifically reserved for important segment level kpis


def _compute_surprise(actual: float, estimate: float) -> float:
    return (actual - estimate) / abs(estimate)


class KPIRetriever:
    def __init__(self) -> None:
        self.va_ch_client = VisAlphaClickHouseClient()

    def convert_kpi_text_to_metadata(
        self, gbi_id: int, kpi_texts: List[KPIText]
    ) -> Dict[int, KPIMetadata]:
        # Assumes all kpis will be for the same gbi_id
        kpis = self.get_all_company_kpis(gbi_id=gbi_id)

        relevant_pids = [kpi_text.id for kpi_text in kpi_texts]
        kpi_lookup: Dict[int, KPIMetadata] = {}
        for kpi in kpis:
            if kpi.pid in relevant_pids:
                kpi_lookup[kpi.pid] = kpi
        return kpi_lookup

    def _merge_actuals_and_estimates_data(
        self,
        kpis: List[KPIMetadata],
        actuals_kpi_data: Dict[int, List[KPIDatapoint]],
        estimates_kpi_data: Dict[int, List[KPIDatapoint]],
    ) -> Dict[str, List[KPIInstance]]:
        kpi_instances_dict: Dict[str, List[KPIInstance]] = {}
        for kpi in kpis:
            pid = kpi.pid
            kpi_name = kpi.name
            actuals = actuals_kpi_data[pid]
            estimates = estimates_kpi_data[pid]
            kpi_instances: List[KPIInstance] = []

            for est_data in estimates:
                kpi_instance = KPIInstance(
                    name=kpi.name,
                    quarter=est_data.quarter,
                    year=est_data.year,
                    estimate=est_data.value,
                    unit=est_data.unit,
                    long_unit=est_data.long_unit,
                    currency=est_data.currency,
                )
                found_matching_actuals = False

                for acc_data in actuals:
                    if (est_data.year, est_data.quarter) == (acc_data.year, acc_data.quarter):
                        kpi_instance.surprise = _compute_surprise(acc_data.value, est_data.value)
                        kpi_instance.actual = acc_data.value
                        found_matching_actuals = True

                # TODO: Current workaround for showing actual data until we get a live dataprovider
                if not found_matching_actuals:
                    kpi_instance.actual = est_data.value
                    kpi_instance.surprise = 0.0

                kpi_instances.append(kpi_instance)
            kpi_instances_dict[kpi_name] = kpi_instances
        return kpi_instances_dict

    def _get_KPI_historical_results_via_clickhouse(
        self,
        gbi: int,
        kpis: List[KPIMetadata],
        dataset: VisAlphaDataset,
        num_prev_quarters: int,
        num_future_quarters: int,
        quarter: Optional[int] = None,
        year: Optional[int] = None,
        max_date: Optional[datetime.datetime] = None,
    ) -> Dict[str, List[KPIInstance]]:
        actuals_kpi_data = self.va_ch_client.fetch_data_for_kpis(
            gbi_id=gbi,
            kpis=kpis,
            quarter=quarter,
            year=year,
            num_prev_quarters=num_prev_quarters,
            num_future_quarters=num_future_quarters,
            dataset=dataset,
            starting_date=max_date,
        )
        estimates_kpi_data = self.va_ch_client.fetch_data_for_kpis(
            gbi_id=gbi,
            kpis=kpis,
            quarter=quarter,
            year=year,
            num_prev_quarters=num_prev_quarters,
            num_future_quarters=num_future_quarters,
            estimate=True,
            dataset=dataset,
            starting_date=max_date,
        )

        # TODO: Currently removing the check for actuals due to lack of live data, will always
        # by 0 for current/future only queries
        if len(estimates_kpi_data) == 0:
            logger.error(f"Couldn't find KPI data for {gbi} for the following kpis:\n{kpis}")
            return {}

        return self._merge_actuals_and_estimates_data(kpis, actuals_kpi_data, estimates_kpi_data)

    def get_all_company_kpis(self, gbi_id: int) -> List[KPIMetadata]:
        return self.va_ch_client.get_company_kpis_for_gbi(gbi_id=gbi_id)

    def get_all_company_kpis_current_year_quarter_via_clickhouse(
        self, gbi_id: int
    ) -> Tuple[List[KPIMetadata], Dict[int, List[KPIDatapoint]]]:
        kpis = self.get_all_company_kpis(gbi_id=gbi_id)
        kpi_data = self.va_ch_client.fetch_data_for_kpis(
            gbi_id=gbi_id, kpis=kpis, starting_date=get_now_utc(), estimate=True
        )
        return kpis, kpi_data

    def get_kpis_by_year_quarter_via_clickhouse(
        self,
        gbi_id: int,
        kpis: List[KPIMetadata],
        dataset: VisAlphaDataset = VisAlphaDataset.COMPANY_DATASET,
        num_prev_quarters: int = 0,
        num_future_quarters: int = 7,
        year: Optional[int] = None,
        quarter: Optional[int] = None,
        starting_date: Optional[datetime.datetime] = None,
    ) -> Dict[str, List[KPIInstance]]:
        """
        Given a starting fiscal year and quarter, this method will return a dictionary containing information on
        key performance indicators for a given company specified by gbi_id. If nothing is passed in, this method
        will return year-quarter data for all KPIs listed in Visible Alpha's Key Value table for the given company.

        The output dictionary's key will indicate the specific name of the KPI while the value is a list of KPIInstance
        instances containing the KPI information for a given year-quarter.

        The amount of year-quarter datapoints collected is determined by num_past_earnings_to_retrieve
        which indicates how many quarters back to go from the initial year and quarter inputted.
        If the inputted year-quarter is in the future, estimated data will be provided for
        year-quarters that have yet to arrive, this is indicated via KPIInstance.is_estimate.
        """
        if not quarter and not year:
            assert (
                starting_date is not None
            ), "Must pass in max_date if no year or quarter specified"

        if not starting_date:
            assert year is not None, "Must pass in year and quarter if max_date is not specified"
            assert quarter is not None, "Must pass in year and quarter if max_date is not specified"

        all_historical_kpis_dict: Dict[str, List[KPIInstance]] = {}

        all_historical_kpis_dict = self._get_KPI_historical_results_via_clickhouse(
            gbi=gbi_id,
            kpis=kpis,
            dataset=dataset,
            num_prev_quarters=num_prev_quarters,
            num_future_quarters=num_future_quarters,
            quarter=quarter,
            year=year,
            max_date=starting_date,
        )

        # If we could not find the standard KPIs, then it is safe to assume
        # visible alpha does not support this company, no point in checking for
        # important company kpis
        if not all_historical_kpis_dict:
            return {}

        # Make sure the results are sorted desc by (year, quarter)
        for name in all_historical_kpis_dict.keys():
            all_historical_kpis_dict[name] = sorted(
                all_historical_kpis_dict[name],
                key=lambda d: (d.year, d.quarter),
            )
        return all_historical_kpis_dict
