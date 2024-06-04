import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from gbi_common_py_utils.utils.clickhouse_base import ClickhouseBase
from gbi_common_py_utils.utils.environment import PROD_TAG

from agent_service.utils.environment import EnvironmentUtils

logger = logging.getLogger(__name__)


class VisAlphaDataset(Enum):
    STANDARD_DATASET = "SD"  # Standard Dataset
    COMPANY_DATASET = "CD"  # Company Dataset


class Clickhouse(ClickhouseBase):
    def __init__(self, environment: Optional[str] = None):
        environment: str = environment or EnvironmentUtils.aws_ssm_prefix
        super().__init__(environment)
        self._env = environment

        ################################################################################################
        # Visible Alpha
        ################################################################################################

        # For VA data since we do not have anything on prod we will want to
        # always override on DEV then use _actual_env to figure out whether
        # we are dealing with dev or alpha gbi_ids
        self._actual_env = EnvironmentUtils.aws_ssm_prefix
        self._env = environment

    def get_cid_for_gbi_ids(self, gbi_ids: List[int]) -> Dict[int, str]:
        if self._actual_env.lower() == PROD_TAG.lower():
            return self._get_cid_for_gbi_ids_alpha(gbi_ids)
        else:
            return self._get_cid_for_gbi_ids_dev(gbi_ids)

    def _get_cid_for_gbi_ids_dev(self, gbi_ids: List[int]) -> Dict[int, str]:
        sql = """
            SELECT x.gbi_id, x.cid
            FROM nlp_service.gbi_cid_lookup_dev x
            WHERE x.gbi_id IN %(gbi_ids)s
        """
        res = self.generic_read(sql, {"gbi_ids": gbi_ids})
        output = {item["gbi_id"]: str(item["cid"]) for item in res}
        return output

    def _get_cid_for_gbi_ids_alpha(self, gbi_ids: List[int]) -> Dict[int, str]:
        sql = """
            SELECT x.gbi_id, x.cid
            FROM nlp_service.gbi_cid_lookup_alpha x
            WHERE x.gbi_id IN %(gbi_ids)s
        """
        res = self.generic_read(sql, {"gbi_ids": gbi_ids})
        output = {item["gbi_id"]: str(item["cid"]) for item in res}
        return output

    def get_company_data_kpis(
        self, cid: str, vid: Optional[str] = None, pids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        vid_clause = "AND ViewId = %(vid)s" if vid is not None else ""
        pid_caluse = "AND ParameterId IN %(pids)s" if pids is not None else ""

        params: Dict[str, Any] = {"cid": cid}
        if vid is not None:
            params["vid"] = vid
        if pids is not None:
            params["pids"] = pids

        sql = f"""
        SELECT DISTINCT
            VACompanyId,
            ParameterId,
            ParameterName
        FROM visible_s3_queue.NormalizedCompanyMeta
        WHERE
            VACompanyId = %(cid)s
            AND ParameterName IS NOT NULL AND TRIM(ParameterName) != ''
            {vid_clause}
            {pid_caluse}
        """
        res = self.generic_read(sql, params=params)
        return res

    def get_kpi_actual_values_for_company_data(
        self, cid: str, pids: List[str], periods: List[str], use_fiscal_year: bool
    ) -> List[Dict[str, Any]]:
        # periods can be generated using visible_alpha.generate_periods(...)
        calendar_type = "Period" if use_fiscal_year else "CalendarPeriod"
        sql = f"""
            WITH actual_data AS (
            SELECT DISTINCT ON (VACompanyId, ParameterId, {calendar_type})
                VACompanyId,
                ParameterId,
                {calendar_type} AS Period,
                Value
            FROM visible_s3_queue.NormalizedVAActualsData
            WHERE
                VACompanyId = %(cid)s
                AND ParameterId IN %(pids)s
                AND {calendar_type} IN %(periods)s
            ),
            company_meta AS (
                SELECT DISTINCT ON (VACompanyId, ParameterId)
                    VACompanyId,
                    ParameterId,
                    ParameterName,
                    UnitId,
                    Currency
                FROM visible_s3_queue.NormalizedCompanyMeta
                WHERE VACompanyId = %(cid)s
                ORDER BY VACompanyId, ParameterId, ViewId
            )
            SELECT
                ad.VACompanyId,
                ad.ParameterId,
                cm.ParameterName,
                ad.Period,
                ad.Value,
                cm.UnitId,
                cm.Currency
            FROM actual_data ad
            INNER JOIN company_meta cm
                ON ad.ParameterId = cm.ParameterId
                AND ad.VACompanyId = cm.VACompanyId
        """
        res = self.generic_read(sql, {"cid": cid, "pids": pids, "periods": periods})
        return res

    def get_kpi_actual_values_for_standardized_data(
        self, cid: str, pids: List[str], periods: List[str], use_fiscal_year: bool
    ) -> List[Dict[str, Any]]:
        # periods can be generated using visible_alpha.generate_periods(...)
        calendar_type = "Period" if use_fiscal_year else "CalendarPeriod"
        sql = f"""
            WITH actual_data AS (
            SELECT DISTINCT ON (VACompanyId, ParameterId, {calendar_type})
                VACompanyId,
                ParameterId,
                SLIUniversalId,
                {calendar_type} AS Period,
                Value
            FROM visible_s3_queue.StandardizedVAActualsData
            WHERE
                VACompanyId = %(cid)s
                AND SLIUniversalId IN %(pids)s
                AND {calendar_type} IN %(periods)s
            ),
            company_meta AS (
                SELECT DISTINCT ON (VACompanyId, ParameterId)
                    VACompanyId,
                    ParameterId,
                    ParameterName,
                    UnitId,
                    Currency
                FROM visible_s3_queue.StandardizedCompanyMeta
                WHERE VACompanyId = %(cid)s
                ORDER BY VACompanyId, ParameterId, ViewId
            )
            SELECT
                ad.VACompanyId,
                ad.SLIUniversalId AS ParameterId,
                cm.ParameterName,
                ad.Period,
                ad.Value,
                cm.UnitId,
                cm.Currency
            FROM actual_data ad
            INNER JOIN company_meta cm
                ON ad.ParameterId = cm.ParameterId
                AND ad.VACompanyId = cm.VACompanyId
        """
        res = self.generic_read(sql, {"cid": cid, "pids": pids, "periods": periods})
        return res

    def get_kpi_estimate_values_for_company_data(
        self, cid: str, pids: List[str], periods: List[str], use_fiscal_year: bool
    ) -> List[Dict[str, Any]]:
        # periods can be generated using visible_alpha.generate_periods(...)
        calendar_type = "Period" if use_fiscal_year else "CalendarPeriod"
        sql = f"""
            WITH
            consensus_data AS (
                SELECT DISTINCT ON (VACompanyId, ParameterId, {calendar_type})
                    VACompanyId,
                    ParameterId,
                    {calendar_type} AS Period,
                    Value,
                    RevisionDate
                FROM visible_s3_queue.NormalizedConsensusData
                WHERE
                    VACompanyId = %(cid)s
                    AND ParameterId IN %(pids)s
                    AND {calendar_type} IN %(periods)s
                    AND Value IS NOT NULL
                    AND TRIM(Value) != ''
                ORDER BY VACompanyId, ParameterId, {calendar_type}, RevisionDate DESC
            ),
            company_meta AS (
                SELECT DISTINCT ON (VACompanyId, ParameterId)
                    VACompanyId,
                    ParameterId,
                    ParameterName,
                    UnitId,
                    Currency
                FROM visible_s3_queue.NormalizedCompanyMeta
                WHERE VACompanyId = %(cid)s
                ORDER BY VACompanyId, ParameterId, ViewId
            )
            SELECT
                cd.VACompanyId,
                cd.ParameterId,
                cm.ParameterName,
                cd.Period,
                cd.Value,
                cm.UnitId,
                cm.Currency
            FROM consensus_data cd
            INNER JOIN company_meta cm ON cd.VACompanyId = cm.VACompanyId AND cd.ParameterId = cm.ParameterId
        """

        res = self.generic_read(sql, {"cid": cid, "pids": pids, "periods": periods})
        return res

    def get_kpi_estimate_values_for_standardized_data(
        self, cid: str, pids: List[str], periods: List[str], use_fiscal_year: bool
    ) -> List[Dict[str, Any]]:
        # periods can be generated using visible_alpha.generate_periods(...)
        calendar_type = "Period" if use_fiscal_year else "CalendarPeriod"
        sql = f"""
            WITH
            consensus_data AS (
                SELECT DISTINCT ON (VACompanyId, ParameterId, {calendar_type})
                    VACompanyId,
                    ParameterId,
                    SLIUniversalId,
                    {calendar_type} AS Period,
                    Value,
                    RevisionDate
                FROM visible_s3_queue.StandardizedConsensusData
                WHERE
                    VACompanyId = %(cid)s
                    AND SLIUniversalId IN %(pids)s
                    AND {calendar_type} IN %(periods)s
                    AND Value IS NOT NULL
                    AND TRIM(Value) != ''
                ORDER BY VACompanyId, ParameterId, {calendar_type}, RevisionDate DESC
            ),
            company_meta AS (
                SELECT DISTINCT ON (VACompanyId, ParameterId)
                    VACompanyId,
                    ParameterId,
                    ParameterName,
                    UnitId,
                    Currency
                FROM visible_s3_queue.StandardizedCompanyMeta
                WHERE VACompanyId = %(cid)s
                ORDER BY VACompanyId, ParameterId, ViewId
            )
            SELECT
                cd.VACompanyId,
                cd.SLIUniversalId AS ParameterId,
                cm.ParameterName,
                cd.Period,
                cd.Value,
                cm.UnitId,
                cm.Currency
            FROM consensus_data cd
            INNER JOIN company_meta cm ON cd.VACompanyId = cm.VACompanyId AND cd.ParameterId = cm.ParameterId
        """
        res = self.generic_read(
            sql, {"cid": cid, "pids": pids, "periods": periods, "calendar_type": calendar_type}
        )
        return res

    def get_kpi_values_for_cid(
        self,
        cid: str,
        pids: List[str],
        periods: List[str],
        dataset: VisAlphaDataset,
        use_fiscal_year: bool,
        estimate: bool,
    ) -> List[Dict[str, Any]]:
        if dataset == VisAlphaDataset.COMPANY_DATASET:
            if estimate:
                return self.get_kpi_estimate_values_for_company_data(
                    cid, pids, periods, use_fiscal_year
                )
            else:
                return self.get_kpi_actual_values_for_company_data(
                    cid, pids, periods, use_fiscal_year
                )
        elif dataset == VisAlphaDataset.STANDARD_DATASET:
            if estimate:
                return self.get_kpi_estimate_values_for_standardized_data(
                    cid, pids, periods, use_fiscal_year
                )
            else:
                return self.get_kpi_actual_values_for_standardized_data(
                    cid, pids, periods, use_fiscal_year
                )
        else:
            raise ValueError("Unsupported dataset")
