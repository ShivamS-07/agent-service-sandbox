import datetime
import json
import logging
import traceback
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import backoff
import clickhouse_connect
from clickhouse_connect.driver.asyncclient import AsyncClient
from clickhouse_connect.driver.query import QueryResult
from gbi_common_py_utils.utils.clickhouse_base import (
    ClickhouseConnectionConfig,
    get_clickhouse_connection_config,
)
from gbi_common_py_utils.utils.environment import (
    PROD_TAG,
    get_non_local_environment_tag,
)

from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.logs import async_perf_logger

logger = logging.getLogger(__name__)


class VisAlphaDataset(Enum):
    STANDARD_DATASET = "SD"  # Standard Dataset
    COMPANY_DATASET = "CD"  # Company Dataset


def get_short_service_version(service_version: str) -> str:
    index = service_version.find(":") + 1
    return service_version[index:]


@backoff.on_exception(
    backoff.expo,
    (
        clickhouse_connect.driver.exceptions.OperationalError,
        clickhouse_connect.driver.exceptions.InternalError,
        clickhouse_connect.driver.exceptions.DatabaseError,
        ValueError,
    ),
    max_tries=5,
    jitter=backoff.full_jitter,
)
async def get_client(clickhouse_connection_config: ClickhouseConnectionConfig) -> AsyncClient:
    return await clickhouse_connect.get_async_client(**clickhouse_connection_config._asdict())


class AsyncClickhouseBase:
    def __init__(
        self,
        environment: Optional[str] = None,
        clickhouse_connection_config: Optional[ClickhouseConnectionConfig] = None,
    ):
        self.environment = (
            environment.lower() if environment else get_non_local_environment_tag().lower()
        )
        self.clickhouse_connection_config: ClickhouseConnectionConfig = (
            get_clickhouse_connection_config(
                environment=self.environment,
                clickhouse_connection_config=clickhouse_connection_config,
            )
        )
        self.client: Optional[AsyncClient] = None

    async def get_or_create_client(self) -> AsyncClient:
        # only create connection in the beginning
        if self.client is None:
            self.client = await get_client(self.clickhouse_connection_config)
        return self.client

    async def generic_read(
        self, sql: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        client = await self.get_or_create_client()
        query_result: QueryResult = await client.query(sql, parameters=params)
        column_names = query_result.column_names
        result_rows = query_result.result_rows
        result = []
        for row in result_rows:
            to_add = {}
            for i, column_name in enumerate(column_names):
                to_add[column_name] = row[i]
            result.append(to_add)

        return result

    async def multi_row_insert(self, table_name: str, rows: List[Dict[str, Any]]) -> None:
        client = await self.get_or_create_client()
        column_names = list(rows[0].keys())
        data_to_insert = []
        for row in rows:
            tuple_to_add = []
            for column_name in column_names:
                tuple_to_add.append(row[column_name])
            data_to_insert.append(tuple_to_add)

        await client.insert(
            table=table_name,
            data=data_to_insert,
            column_names=column_names,
            settings={"async_insert": 1},
        )


class Clickhouse(AsyncClickhouseBase):
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

    async def get_cid_for_gbi_ids(self, gbi_ids: List[int]) -> Dict[int, str]:
        if self._actual_env.lower() == PROD_TAG.lower():
            return await self._get_cid_for_gbi_ids_alpha(gbi_ids)
        else:
            return await self._get_cid_for_gbi_ids_dev(gbi_ids)

    async def _get_cid_for_gbi_ids_dev(self, gbi_ids: List[int]) -> Dict[int, str]:
        sql = """
            SELECT x.gbi_id, x.cid
            FROM nlp_service.gbi_cid_lookup_dev x
            WHERE x.gbi_id IN %(gbi_ids)s
        """
        res = await self.generic_read(sql, {"gbi_ids": gbi_ids})
        output = {item["gbi_id"]: str(item["cid"]) for item in res}
        return output

    async def _get_cid_for_gbi_ids_alpha(self, gbi_ids: List[int]) -> Dict[int, str]:
        sql = """
            SELECT x.gbi_id, x.cid
            FROM nlp_service.gbi_cid_lookup_alpha x
            WHERE x.gbi_id IN %(gbi_ids)s
        """
        res = await self.generic_read(sql, {"gbi_ids": gbi_ids})
        output = {item["gbi_id"]: str(item["cid"]) for item in res}
        return output

    async def get_company_data_kpis(
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
        res = await self.generic_read(sql, params=params)
        return res

    async def get_company_data_kpis_for_multiple_cids(
        self, cids: List[str]
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT DISTINCT
                VACompanyId,
                ParameterId,
                ParameterName
            FROM visible_s3_queue.NormalizedCompanyMeta
            WHERE VACompanyId IN %(cids)s AND ParameterName IS NOT NULL AND TRIM(ParameterName) != ''
        """
        res = await self.generic_read(sql, params={"cids": cids})
        return res

    async def get_kpi_actual_values_for_company_data(
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
        res = await self.generic_read(sql, {"cid": cid, "pids": pids, "periods": periods})
        return res

    async def get_kpi_actual_values_for_standardized_data(
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
        res = await self.generic_read(sql, {"cid": cid, "pids": pids, "periods": periods})
        return res

    async def get_kpi_estimate_values_for_company_data(
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

        res = await self.generic_read(sql, {"cid": cid, "pids": pids, "periods": periods})
        return res

    async def get_kpi_estimate_values_for_standardized_data(
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
        res = await self.generic_read(
            sql, {"cid": cid, "pids": pids, "periods": periods, "calendar_type": calendar_type}
        )
        return res

    async def get_kpi_values_for_cid(
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
                return await self.get_kpi_estimate_values_for_company_data(
                    cid, pids, periods, use_fiscal_year
                )
            else:
                return await self.get_kpi_actual_values_for_company_data(
                    cid, pids, periods, use_fiscal_year
                )
        elif dataset == VisAlphaDataset.STANDARD_DATASET:
            if estimate:
                return await self.get_kpi_estimate_values_for_standardized_data(
                    cid, pids, periods, use_fiscal_year
                )
            else:
                return await self.get_kpi_actual_values_for_standardized_data(
                    cid, pids, periods, use_fiscal_year
                )
        else:
            raise ValueError("Unsupported dataset")

    async def get_kpi_estimate_values_for_company_data_for_multiple_cids(
        self, cid_pids_dict: Dict[str, List[int]], periods: List[str], use_fiscal_year: bool
    ) -> List[Dict[str, Any]]:
        # periods can be generated using visible_alpha.generate_periods(...)
        calendar_type = "Period" if use_fiscal_year else "CalendarPeriod"
        cid_pid_pairwise_list = [
            (company_id, parameter_id)
            for company_id, parameters in cid_pids_dict.items()
            for parameter_id in parameters
        ]
        cids = list(cid_pids_dict.keys())
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
                    (VACompanyId, ParameterId) IN %(cid_pid_pairwise_list)s
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
                WHERE VACompanyId IN %(cids)s
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
        res = await self.generic_read(
            sql, {"cids": cids, "cid_pid_pairwise_list": cid_pid_pairwise_list, "periods": periods}
        )
        return res

    async def get_kpi_actual_values_for_company_data_for_multiple_cids(
        self, cid_pids_dict: Dict[str, List[int]], periods: List[str], use_fiscal_year: bool
    ) -> List[Dict[str, Any]]:
        # periods can be generated using visible_alpha.generate_periods(...)
        calendar_type = "Period" if use_fiscal_year else "CalendarPeriod"
        cid_pid_pairwise_list = [
            (company_id, parameter_id)
            for company_id, parameters in cid_pids_dict.items()
            for parameter_id in parameters
        ]
        cids = list(cid_pids_dict.keys())
        sql = f"""
            WITH actual_data AS (
            SELECT DISTINCT ON (VACompanyId, ParameterId, {calendar_type})
                VACompanyId,
                ParameterId,
                {calendar_type} AS Period,
                Value
            FROM visible_s3_queue.NormalizedVAActualsData
            WHERE
                (VACompanyId, ParameterId) IN %(cid_pid_pairwise_list)s
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
                WHERE VACompanyId IN %(cids)s
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
        res = await self.generic_read(
            sql, {"cids": cids, "cid_pid_pairwise_list": cid_pid_pairwise_list, "periods": periods}
        )
        return res

    async def get_company_kpi_values_for_cids(
        self,
        cid_pids_dict: Dict[str, List[int]],
        periods: List[str],
        use_fiscal_year: bool,
        estimate: bool,
    ) -> List[Dict[str, Any]]:
        """
        This is only meant for handling company specific line items
        """
        if estimate:
            return await self.get_kpi_estimate_values_for_company_data_for_multiple_cids(
                cid_pids_dict, periods, use_fiscal_year
            )
        else:
            return await self.get_kpi_actual_values_for_company_data_for_multiple_cids(
                cid_pids_dict, periods, use_fiscal_year
            )

    ################################################################################################
    # Embeddings
    ################################################################################################
    async def sort_news_topics_via_embeddings(
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
            result = await self.generic_read(sql, params=parameters)
            return [str(row["topic_id"]) for row in result]
        except Exception as e:
            logger.exception(e)
            return []

    ################################################################################################
    # Agent Debug Info
    ################################################################################################
    @async_perf_logger
    async def get_agent_debug_plan_selections(self, agent_id: str) -> List[Dict[str, Any]]:
        sql = """
            SELECT plans, selection_str, selection, plan_id, service_version, start_time_utc,
            end_time_utc, duration_seconds
            FROM agent.plan_selections
            WHERE agent_id = %(agent_id)s
            ORDER BY end_time_utc ASC
            """
        res: List[Dict[str, Any]] = []
        rows = await self.generic_read(sql, {"agent_id": agent_id})
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

    @async_perf_logger
    async def get_agent_debug_plans(self, agent_id: str) -> List[Dict[str, Any]]:
        sql = """
        SELECT execution_plan, action, model_id, plan_str, plan_id, error_msg, service_version,
        start_time_utc, end_time_utc, duration_seconds, sample_plans
        FROM agent.plans
        WHERE agent_id = %(agent_id)s
        ORDER BY end_time_utc ASC
        """
        rows = await self.generic_read(sql, {"agent_id": agent_id})
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

    @async_perf_logger
    async def get_agent_debug_tool_calls(self, agent_id: str) -> Tuple[Dict[str, Any], str]:
        sql = """
        SELECT  plan_id, plan_run_id, task_id, tool_name, start_time_utc,
        end_time_utc, service_version, duration_seconds, error_msg, replay_id, debug_info,
        pod_name
        FROM agent.tool_calls
        WHERE agent_id = %(agent_id)s
        ORDER BY end_time_utc ASC
        """
        rows = await self.generic_read(sql, {"agent_id": agent_id})
        res: Dict[str, Any] = OrderedDict()
        tz = datetime.timezone.utc
        pod_name = ""
        for row in rows:
            plan_run_id = row["plan_run_id"]
            if not pod_name:
                pod_name = row["pod_name"]
            if plan_run_id not in res:
                res[plan_run_id] = OrderedDict()
            tool_name = row["tool_name"]
            row["start_time_utc"] = row["start_time_utc"].replace(tzinfo=tz).isoformat()
            row["end_time_utc"] = row["end_time_utc"].replace(tzinfo=tz).isoformat()
            env_upper = self._env.upper()
            row["replay_command"] = (
                f"ENVIRONMENT={env_upper} pipenv run python run_plan_task.py "
                f"--env {env_upper} --replay-id {row['replay_id']}"
            )
            row[f"args_{row['replay_id']}"] = {}
            row[f"result_{row['replay_id']}"] = {}
            if row["debug_info"]:
                row["debug_info"] = json.loads(row["debug_info"])
            res[plan_run_id][f"{tool_name}_{row['task_id']}"] = row
        return (res, pod_name)

    @async_perf_logger
    async def get_plan_run_debug_tool_calls(self, plan_run_id: str) -> List[Dict]:
        sql = """
        SELECT task_id, tool_name, args, start_time_utc, end_time_utc, duration_seconds
        FROM agent.tool_calls
        WHERE plan_run_id = %(plan_run_id)s
        ORDER BY end_time_utc ASC
        """
        return await self.generic_read(sql, {"plan_run_id": plan_run_id})

    @async_perf_logger
    async def get_plan_run_debug_prompt_infos(self, plan_run_id: str) -> Dict[str, List[Dict]]:
        # `argMax(args, val)` is a ClickHouse function that returns the value of `args` where `val`
        # is the maximum value in the group.

        sql = """
            SELECT task_id, main_prompt_name,
                argMax(sys_prompt_name, num_output_tokens) AS sys_prompt_name,
                argMax(model_id, num_output_tokens) AS gpt_model,
                argMax(main_prompt, num_output_tokens) AS main_prompt_example,
                argMax(sys_prompt, num_output_tokens) AS sys_prompt_example,
                argMax(`result`, num_output_tokens) AS gpt_response_example,
                count(*) AS num_calls,
                sum(num_input_tokens) AS total_num_input_tokens,
                sum(num_output_tokens) AS total_num_output_tokens,
                round(sum(cost_usd), 2) AS total_cost_usd,
                max(`timestamp`) - min(`timestamp`) AS duration_seconds
            FROM llm.queries q
            WHERE plan_run_id = %(plan_run_id)s
            GROUP BY task_id, main_prompt_name
            ORDER BY total_cost_usd DESC
        """
        rows = await self.generic_read(sql, {"plan_run_id": plan_run_id})
        output: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            output[row["task_id"]].append(row)
        return output

    @async_perf_logger
    async def get_agent_debug_cost_info(self, agent_id: str) -> Dict[str, Any]:
        total_cost_sql = """select round(sum(cost_usd), 2) as total_cost_usd from llm.queries q
        where agent_id =  %(agent_id)s"""
        total_cost_rows = await self.generic_read(total_cost_sql, {"agent_id": agent_id})
        total_cost = total_cost_rows[0]["total_cost_usd"]
        detailed_breakdown_sql = """select sum(num_input_tokens) as total_input_tokens,
        sum(num_output_tokens) as total_output_tokens, round(sum(cost_usd), 2) as total_cost_usd,
        main_prompt_name, model_id from llm.queries q where agent_id = %(agent_id)s group by
        main_prompt_name, model_id order by total_cost_usd desc"""
        detailed_breakdown_rows = await self.generic_read(
            detailed_breakdown_sql, {"agent_id": agent_id}
        )
        detailed_breakdown_dict = {}
        for row in detailed_breakdown_rows:
            detailed_breakdown_dict[f"prompt_id={row['main_prompt_name']}"] = row
        model_breakdown_sql = """select round(sum(cost_usd), 2) as total_cost_usd, model_id,
        sum(num_input_tokens) as total_input_tokens, sum(num_output_tokens) as total_output_tokens
        from llm.queries q  where agent_id = %(agent_id)s group by model_id order by
        total_cost_usd desc"""
        result = {}
        result["detailed_breakdown"] = detailed_breakdown_dict
        result["total_cost_usd"] = total_cost
        model_breakdown_rows = await self.generic_read(model_breakdown_sql, {"agent_id": agent_id})
        model_breakdown_dict = {}
        for row in model_breakdown_rows:
            model_breakdown_dict[f"model_id={row['model_id']}"] = row
            del row["model_id"]
        result["model_breakdown"] = model_breakdown_dict
        return result

    @async_perf_logger
    async def get_agent_debug_gpt_service_info(self, agent_id: str) -> Dict[str, Any]:
        sql = """
        select count(*) as num_queries, min(client_timestamp_utc) as first_request_sent_utc,
        max(client_timestamp_utc) as last_request_sent_utc,
        max(response_timestamp_utc) as last_request_completed_utc,
        round(median(duration_seconds), 2) as median_duration_seconds,
        round(median(wait_time_seconds), 2) as median_wait_time_seconds,
        round(max(wait_time_seconds), 2) as max_wait_time_seconds,
        round(max(duration_seconds), 2) as max_duration_seconds,
        model_id,
        main_prompt_name
        from llm.gpt_service_requests gsr
        where agent_id =  %(agent_id)s
        group by main_prompt_name, model_id
        order by num_queries desc
        """
        result = {}
        try:
            rows = await self.generic_read(sql=sql, params={"agent_id": agent_id})
            for row in rows:
                result[f"prompt_id={row['main_prompt_name']}"] = row
                del row["main_prompt_name"]
        except Exception:
            logger.info(f"Unable to get gpt_service_info for {agent_id=}: {traceback.format_exc()}")
        return result

    @async_perf_logger
    async def get_debug_tool_args(self, replay_id: str) -> str:
        sql = """
        select args from agent.tool_calls
        where replay_id =  %(replay_id)s
        """
        res = await self.generic_read(sql, {"replay_id": replay_id})
        if not res:
            return ""
        return res[0]["args"]

    @async_perf_logger
    async def get_debug_tool_result(self, replay_id: str) -> str:
        sql = """
        select result from agent.tool_calls
        where replay_id =  %(replay_id)s
        """
        res = await self.generic_read(sql, {"replay_id": replay_id})
        if not res:
            return ""
        return res[0]["result"]

    @async_perf_logger
    async def get_agent_debug_worker_sqs_log(self, agent_id: str) -> Dict[str, Any]:
        sql = """
        SELECT plan_id, plan_run_id, method, message, send_time_utc,
        wait_time_seconds,  error_msg,start_time_utc, end_time_utc, duration_seconds
        FROM agent.worker_sqs_log
        WHERE agent_id = %(agent_id)s
        ORDER BY end_time_utc ASC
        """
        res: Dict[str, Any] = dict()
        rows = await self.generic_read(sql, {"agent_id": agent_id})
        res["run_execution_plan"] = OrderedDict()
        res["create_execution_plan"] = OrderedDict()
        tz = datetime.timezone.utc
        for row in rows:
            row["start_time_utc"] = row["start_time_utc"].replace(tzinfo=tz).isoformat()
            row["end_time_utc"] = row["end_time_utc"].replace(tzinfo=tz).isoformat()
            row["send_time_utc"] = row["send_time_utc"].replace(tzinfo=tz).isoformat()
            row["message"] = json.loads(row["message"])
            res[row["method"]][row["send_time_utc"]] = row
        return res

    @async_perf_logger
    async def get_agents_cost_info(self, agent_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        res = {}
        try:
            sql = """select sum(num_input_tokens) + sum(num_output_tokens) as num_tokens_used, agent_id
            from llm.queries where agent_id in %(agent_ids)s group by agent_id"""
            rows = await self.generic_read(sql, params={"agent_ids": agent_ids})

            for row in rows:
                res[row["agent_id"]] = [
                    {"label": "Total Tokens Used", "val": row["num_tokens_used"]}
                ]
            return res
        except Exception:
            logger.info(
                f"Error in get_agent_cost_info for agent_ids={agent_ids}: {traceback.format_exc()}"
            )
            return {}

    ################################################################################################
    # Regression Test Run Info
    ################################################################################################

    async def get_info_for_test_suite_run(self, service_version: str) -> Dict[str, Any]:
        sql = """SELECT DISTINCT ON (test_name) test_name, prompt, agent_id, test_suite_id AS test_suite_run_id,
        output, execution_plan, error_msg, warning_msg, execution_finished_at_utc, execution_start_at_utc,
        execution_plan_started_at_utc, execution_plan_finished_at_utc, execution_duration_seconds,
        execution_plan_duration_seconds FROM agent.regression_test
        WHERE service_version = CONCAT('374053208103.dkr.ecr.us-west-2.amazonaws.com/agent-service:',
        %(service_version)s)
        ORDER BY test_name ASC, timestamp DESC"""
        res: Dict[str, Any] = {}
        rows = await self.generic_read(sql, {"service_version": service_version})
        tz = datetime.timezone.utc
        for i, row in enumerate(rows):
            test_name = row.pop("test_name")
            for key in ["output", "execution_plan"]:
                if row[key]:
                    row[key] = json.loads(row[key])
            for key in [
                "execution_plan_started_at_utc",
                "execution_plan_finished_at_utc",
                "execution_start_at_utc",
                "execution_finished_at_utc",
            ]:
                if row[key]:
                    row[key] = row[key].replace(tzinfo=tz).isoformat()
            res[f"{i} test_name={test_name}"] = row
        return res

    async def get_test_suite_runs(self) -> List[Dict[str, Any]]:
        sql = """
        SELECT DISTINCT ON (service_version) MAX(substring(service_version, position(service_version, ':') + 1))
        as service_version, test_suite_id as test_suite_run_id, MAX(timestamp) as timestamp,
        SUM(error_msg != '') AS error_count,
        SUM(warning_msg != '') AS warning_count,
        SUM(error_msg == '' AND warning_msg == '') AS success_count
        FROM agent.regression_test
        GROUP BY test_suite_id
        ORDER BY timestamp DESC
        """
        rows = await self.generic_read(sql)
        tz = datetime.timezone.utc
        for row in rows:
            row["timestamp"] = row["timestamp"].replace(tzinfo=tz).isoformat()
        return rows

    async def get_test_cases(self) -> List[Dict[str, Any]]:
        sql = """
        SELECT test_name as test_case, MAX(timestamp) as timestamp from agent.regression_test
        GROUP BY test_name ORDER BY timestamp DESC
        """
        rows = await self.generic_read(sql)
        tz = datetime.timezone.utc
        for row in rows:
            row["timestamp"] = row["timestamp"].replace(tzinfo=tz).isoformat()
        return rows

    async def get_info_for_test_case(self, test_name: str) -> Dict[str, Any]:
        sql = """
            SELECT prompt, agent_id, output, execution_plan, service_version, error_msg, warning_msg,
            execution_finished_at_utc, execution_start_at_utc, execution_plan_started_at_utc,
            execution_plan_finished_at_utc, execution_duration_seconds, execution_plan_duration_seconds
            FROM agent.regression_test
            WHERE test_name = %(test_name)s
            ORDER BY timestamp DESC
            """
        rows = await self.generic_read(sql, {"test_name": test_name})
        res: Dict[str, Any] = {}
        tz = datetime.timezone.utc
        for row in rows:
            row["output_str"] = row["output"]
            for key in ["output", "execution_plan"]:
                if row[key]:
                    row[key] = json.loads(row[key])
            for key in [
                "execution_plan_started_at_utc",
                "execution_plan_finished_at_utc",
                "execution_start_at_utc",
                "execution_finished_at_utc",
            ]:
                if row[key]:
                    row[key] = row[key].replace(tzinfo=tz).isoformat()
            service_version_short = get_short_service_version(row["service_version"])
            del row["service_version"]
            res[f"version={service_version_short}"] = row
        return res

    ################################################################################################
    # Tool Diff Info
    ################################################################################################

    async def get_io_for_tool_run(
        self, plan_run_id: str, task_id: str, tool_name: str
    ) -> Optional[Tuple[str, str, str, datetime.datetime]]:
        sql = """
        SELECT args, result, debug_info, timestamp
        FROM agent.tool_calls
        WHERE plan_run_id = %(plan_run_id)s AND task_id = %(task_id)s AND tool_name = %(tool_name)s
        """
        rows = await self.generic_read(
            sql, {"plan_run_id": plan_run_id, "task_id": task_id, "tool_name": tool_name}
        )
        if not rows:
            return None
        row = rows[0]
        return row["args"], row["result"], row["debug_info"], row["timestamp"]

    ################################################################################################
    # Manual Event Logging
    ################################################################################################
    async def log_event_manually(
        self,
        event_name: str,
        event_data: Optional[Dict[str, Any]] = None,
        event_namespace: Optional[str] = None,
    ) -> None:
        if event_data is None:
            event_data = {}

        event = {
            "event_data": json.dumps(event_data),
            "event_name": event_name,
            "timestamp": get_now_utc(),
        }
        if event_namespace:
            event["event_namespace"] = event_namespace

        await self.multi_row_insert(table_name="events", rows=[event])

    ################################################################################################
    # Follow up questions
    ################################################################################################
    async def get_task_outputs(
        self, agent_id: str, task_ids: List[str], old_plan_id: str
    ) -> Dict[str, Any]:
        from agent_service.io_type_utils import load_io_type

        sql = """
            SELECT DISTINCT ON (task_id)
                task_id,
                result AS output
            FROM agent.tool_calls
            WHERE
                agent_id = %(agent_id)s
                AND task_id IN %(task_ids)s
                AND plan_id = %(old_plan_id)s
                AND result <> ''
            ORDER BY task_id, end_time_utc DESC
        """
        rows = await self.generic_read(
            sql, params={"agent_id": agent_id, "task_ids": task_ids, "old_plan_id": old_plan_id}
        )
        res = {}
        for row in rows:
            res[row["task_id"]] = load_io_type(row["output"])
        return res

    async def get_task_outputs_from_replay_ids(self, replay_ids: List[str]) -> Dict[str, Any]:
        from agent_service.io_type_utils import load_io_type

        sql = """
            SELECT
                task_id,
                result AS output
            FROM agent.tool_calls
            WHERE
                replay_id IN %(replay_ids)s
                AND result <> ''
        """
        rows = await self.generic_read(sql, params={"replay_ids": replay_ids})
        res = {}
        for row in rows:
            res[row["task_id"]] = load_io_type(row["output"])
        return res

    async def get_task_replay_ids(
        self, agent_id: str, task_ids: List[str], plan_id: str
    ) -> Dict[str, str]:
        """
        Given a list of task id's, return a mapping from task ID to replay ID
        for the most recent row given the agent and plan ids.
        """
        sql = """
            SELECT DISTINCT ON (task_id)
                task_id,
                replay_id
            FROM agent.tool_calls
            WHERE
                agent_id = %(agent_id)s
                AND task_id IN %(task_ids)s
                AND plan_id = %(old_plan_id)s
                AND result <> ''
            ORDER BY task_id, end_time_utc DESC
        """
        rows = await self.generic_read(
            sql, params={"agent_id": agent_id, "task_ids": task_ids, "old_plan_id": plan_id}
        )
        res = {}
        for row in rows:
            res[row["task_id"]] = row["replay_id"]
        return res
