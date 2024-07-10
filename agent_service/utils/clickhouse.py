import datetime
import json
import logging
from collections import OrderedDict
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

    ################################################################################################
    # Embeddings
    ################################################################################################
    def sort_news_topics_via_embeddings(
        self,
        news_topic_ids: List[str],
        embedding_vector: List[float],
        embedding_model_id: str,
        min_created_at: Optional[datetime.datetime] = None,
    ) -> List[str]:
        parameters: Dict[str, Any] = {
            "embedding_model_id": embedding_model_id,
            "embedding_group_id": "stock_news_topic",
            "topic_ids": news_topic_ids,
            "embedding_vector": embedding_vector,
        }

        dt_filter = ""
        if min_created_at is not None:
            dt_filter = "AND parseDateTimeBestEffort(JSONExtractString(metadata, 'created_at')) >= %(min_created_at)s"
            parameters["min_created_at"] = min_created_at

        sql = f"""
            WITH topics AS (
                SELECT DISTINCT ON (topic_id) topic_id, embedding_vector
                FROM embeddings.embeddings e2
                WHERE embedding_model_id = %(embedding_model_id)s
                    AND embedding_group_id = %(embedding_group_id)s
                    AND topic_id IN %(topic_ids)s
                    {dt_filter}
                ORDER BY topic_id, updated_at DESC
            )
            SELECT topic_id
            FROM topics
            ORDER BY cosineDistance(embedding_vector, %(embedding_vector)s) ASC
        """
        # FIXME: `generic_read` has bad performance, so we use `query` directly
        try:
            result = self.clickhouse_client.query(sql, parameters=parameters)
            return [tup[0] for tup in result.result_rows]
        except Exception as e:
            logger.exception(e)
            return []

    ################################################################################################
    # Agent Debug Info
    ################################################################################################
    def get_agent_debug_plan_selections(self, agent_id: str) -> List[Dict[str, Any]]:
        sql = """
            SELECT plans, selection_str, selection, plan_id, service_version, start_time_utc,
            end_time_utc, duration_seconds
            FROM agent.plan_selections
            WHERE agent_id = %(agent_id)s
            ORDER BY end_time_utc DESC
            """
        res: List[Dict[str, Any]] = []
        rows = self.generic_read(sql, {"agent_id": agent_id})
        tz = datetime.timezone.utc
        for row in rows:
            row["start_time_utc"] = row["start_time_utc"].replace(tzinfo=tz).isoformat()
            row["end_time_utc"] = row["end_time_utc"].replace(tzinfo=tz).isoformat()
            row["plans"] = json.loads(row["plans"])
            for i, plan in enumerate(row["plans"]):
                plans_dict = json.loads(plan)
                steps_dict: Dict[str, Any] = OrderedDict()
                for step in plans_dict:
                    steps_dict[f"{step['tool_name']}_{step['tool_task_id']}"] = step
                row["plans"][i] = steps_dict
            res.append(row)
        return res

    def get_agent_debug_plans(self, agent_id: str) -> List[Dict[str, Any]]:
        sql = """
        SELECT execution_plan, action, model_id, plan_str, plan_id, error_msg, service_version,
        start_time_utc, end_time_utc, duration_seconds, sample_plans
        FROM agent.plans
        WHERE agent_id = %(agent_id)s
        ORDER BY end_time_utc DESC
        """
        rows = self.generic_read(sql, {"agent_id": agent_id})
        tz = datetime.timezone.utc
        for row in rows:
            steps_dict: Dict[str, Any] = OrderedDict()
            row["start_time_utc"] = row["start_time_utc"].replace(tzinfo=tz).isoformat()
            row["end_time_utc"] = row["end_time_utc"].replace(tzinfo=tz).isoformat()
            if row["execution_plan"]:
                execution_plan_dict = json.loads(row["execution_plan"])
                for step in execution_plan_dict:
                    steps_dict[f"{step['tool_name']}_{step['tool_task_id']}"] = step
            row["execution_plan"] = steps_dict
        return rows

    def get_agent_debug_tool_calls(self, agent_id: str) -> Dict[str, Any]:
        sql = """
        SELECT  plan_id, plan_run_id, task_id, tool_name, args, result, start_time_utc,
        end_time_utc, service_version, duration_seconds, error_msg, replay_id
        FROM agent.tool_calls
        WHERE agent_id = %(agent_id)s
        ORDER BY end_time_utc DESC
        """
        rows = self.generic_read(sql, {"agent_id": agent_id})
        res: Dict[str, Any] = OrderedDict()
        tz = datetime.timezone.utc
        for row in rows:
            plan_run_id = row["plan_run_id"]
            if plan_run_id not in res:
                res[plan_run_id] = OrderedDict()
            tool_name = row["tool_name"]
            row["start_time_utc"] = row["start_time_utc"].replace(tzinfo=tz).isoformat()
            row["end_time_utc"] = row["end_time_utc"].replace(tzinfo=tz).isoformat()
            row["replay_command"] = (
                f"pipenv run python run_plan_task.py --env {self._env.upper()} --replay-id {row['replay_id']}"
            )
            if row["args"]:
                row["args"] = json.loads(row["args"])
            if row["result"]:
                row["result"] = json.loads(row["result"])
            res[plan_run_id][f"{tool_name}_{row['task_id']}"] = row
        return res

    def get_agent_debug_worker_sqs_log(self, agent_id: str) -> Dict[str, Any]:
        sql = """
        SELECT plan_id, plan_run_id, method, arguments, message, send_time_utc,
        wait_time_seconds,  error_msg,start_time_utc, end_time_utc, duration_seconds
        FROM agent.worker_sqs_log
        WHERE agent_id = %(agent_id)s
        ORDER BY end_time_utc DESC
        """
        res: Dict[str, Any] = dict()
        rows = self.generic_read(sql, {"agent_id": agent_id})
        res["run_execution_plan"] = OrderedDict()
        res["create_execution_plan"] = OrderedDict()
        tz = datetime.timezone.utc
        for row in rows:
            row["start_time_utc"] = row["start_time_utc"].replace(tzinfo=tz).isoformat()
            row["end_time_utc"] = row["end_time_utc"].replace(tzinfo=tz).isoformat()
            row["send_time_utc"] = row["send_time_utc"].replace(tzinfo=tz).isoformat()
            row["message"] = json.loads(row["message"])
            row["arguments"] = json.loads(row["arguments"])
            res[row["method"]][row["send_time_utc"]] = row
        return res
