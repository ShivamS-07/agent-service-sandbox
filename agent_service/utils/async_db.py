import asyncio
import datetime
import json
import logging
import re
import time
import uuid
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from dateutil.parser import parse
from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    get_environment_tag,
)
from psycopg.sql import SQL, Identifier, Placeholder

from agent_service.endpoints.models import (
    AgentFeedback,
    AgentInfo,
    AgentMetadata,
    AgentNotificationEmail,
    AgentOutput,
    AgentQC,
    AgentQCInfo,
    AgentSchedule,
    AgentUserSettingsSetRequest,
    CustomNotification,
    HorizonCriteria,
    HorizonCriteriaOperator,
    Pagination,
    PlanRunStatusInfo,
    QuickThoughts,
    SetAgentFeedBackRequest,
    Status,
    TaskRunStatusInfo,
    TaskStatus,
)
from agent_service.io_type_utils import (
    IOType,
    dump_io_type,
    load_io_type,
    safe_dump_io_type,
)

# Make sure all io_types are registered
from agent_service.io_types import *  # noqa
from agent_service.io_types.graph import GraphOutput
from agent_service.io_types.output import Output
from agent_service.io_types.table import TableOutput
from agent_service.io_types.text import Text, TextOutput
from agent_service.planner.planner_types import (
    ExecutionPlan,
    OutputWithID,
    PlanStatus,
    RunMetadata,
)
from agent_service.types import (
    AgentUserSettings,
    AgentUserSettingsSource,
    ChatContext,
    Message,
    Notification,
    PlanRunContext,
)
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.async_utils import run_async_background
from agent_service.utils.boosted_pg import BoostedPG, InsertToTableArgs
from agent_service.utils.cache_utils import RedisCacheBackend
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import Postgres, SyncBoostedPG
from agent_service.utils.prompt_template import PromptTemplate, UserOrganization
from agent_service.utils.sidebar_sections import SidebarSection

logger = logging.getLogger(__name__)


class AsyncDB:
    def __init__(self, pg: BoostedPG):
        self.pg = pg

    async def generic_read(self, sql: str, params: Optional[Any] = None) -> List[Dict]:
        return await self.pg.generic_read(sql, params)

    async def generic_write(self, sql: str, params: Optional[Any] = None) -> None:
        return await self.pg.generic_write(sql, params)

    async def get_prev_outputs_for_agent_plan(
        self,
        agent_id: str,
        plan_id: str,
        latest_plan_run_id: str,
        cutoff_dt: Optional[datetime.datetime],
    ) -> Optional[Tuple[List[IOType], datetime.datetime]]:
        """
        Returns the prior list of outputs for a plan, as well as the date the outputs were created.
        """
        # we could do this in one larger sql, but decomposing it to make sure
        # get_prev_outputs_for_agent_plan() and get_previous_plan_run() have similar
        # implementations
        # in the future we could harmonize the better

        prev_plan_run_id, plan_run_created_at = await self.get_previous_plan_run(
            agent_id=agent_id,
            plan_id=plan_id,
            latest_plan_run_id=latest_plan_run_id,
            cutoff_dt=cutoff_dt,
        )

        if not prev_plan_run_id:
            return None

        # I suspect this '"output" NOTNULL' is hiding the case where an error occured in prev run
        # but that probably means we should have reverted to an even earlier run
        # that logic will be handled in get_previous_plan_run()
        sql = """
        SELECT plan_run_id,
          MAX(ao.created_at) AS prev_date,
          ARRAY_AGG(ao.output ORDER BY ao.created_at) AS outputs
        FROM agent.agent_outputs ao
        WHERE agent_id = %(agent_id)s AND "output" NOTNULL AND is_intermediate = FALSE
                AND NOT ao.deleted
                AND plan_id = %(plan_id)s
                AND plan_run_id = %(prev_plan_run_id)s
        GROUP BY plan_run_id
        LIMIT 1
        """
        rows = await self.pg.generic_read(
            sql,
            {"agent_id": agent_id, "plan_id": plan_id, "prev_plan_run_id": prev_plan_run_id},
        )
        if not rows:
            return None
        row = rows[0]
        return ([load_io_type(output) for output in row["outputs"]], row["prev_date"])

    async def get_previous_plan_run(
        self,
        agent_id: str,
        plan_id: str,
        latest_plan_run_id: str,
        cutoff_dt: Optional[datetime.datetime],
    ) -> Tuple[Optional[str], Optional[datetime.datetime]]:
        """
        Returns the last plan run for the agent before the latest_run, along with the timestamp for that run
        """
        date_filter = ""
        if cutoff_dt:
            date_filter = " AND created_at < %(cutoff_dt)s"

        # TODO should we be filtering to plan_runs_that are
        # actually in the completed state (not error)
        sql = f"""
        SELECT plan_run_id, created_at FROM agent.plan_runs
        WHERE plan_run_id != %(latest_plan_run_id)s
            AND agent_id = %(agent_id)s
            AND plan_id = %(plan_id)s{date_filter}
        ORDER BY created_at DESC
        LIMIT 1
        """

        rows = await self.pg.generic_read(
            sql,
            {
                "agent_id": agent_id,
                "plan_id": plan_id,
                "latest_plan_run_id": latest_plan_run_id,
                "cutoff_dt": cutoff_dt,
            },
        )
        if rows:
            return rows[0]["plan_run_id"], rows[0]["created_at"]
        else:
            return None, None

    async def get_agent_outputs(
        self,
        agent_id: str,
        plan_run_id: Optional[str] = None,
        cache: Optional[RedisCacheBackend] = None,
    ) -> List[AgentOutput]:
        """
        if `plan_run_id` is None, get the latest run's outputs
        """
        if cache:
            return await self.get_agent_outputs_cache(cache, agent_id, plan_run_id)
        else:
            return await self.get_agent_outputs_no_cache(agent_id, plan_run_id)

    async def get_agent_outputs_data_from_db(
        self,
        agent_id: str,
        include_output: bool,
        plan_run_id: Optional[str] = None,
        output_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        if `plan_run_id` is None, get the latest run's outputs
        """

        if plan_run_id:
            where_clause = """
            ao.plan_run_id = %(plan_run_id)s
            AND NOT ao.deleted
            AND ao.output NOTNULL AND ao.is_intermediate = FALSE
            """
            params = {"plan_run_id": plan_run_id}
        elif output_id:
            where_clause = """
            ao.output_id = %(output_id)s
            AND NOT ao.deleted
            """
            params = {"output_id": output_id}
        else:
            where_clause = """
                ao.plan_run_id IN (
                        SELECT plan_run_id FROM agent.agent_outputs
                        WHERE agent_id = %(agent_id)s AND "output" NOTNULL AND is_intermediate = FALSE
                        ORDER BY created_at DESC LIMIT 1
                    )
                AND NOT ao.deleted
            """
            params = {"agent_id": agent_id}

        columns = [
            "ao.plan_id::VARCHAR",
            "ao.output_id::VARCHAR",
            "ao.plan_run_id::VARCHAR",
            "ao.task_id::VARCHAR",
            "ao.is_intermediate",
            "ao.live_plan_output",
            "ao.created_at",
            "pr.shared",
            "pr.run_metadata",
            "ep.plan",
            "ep.locked_tasks",
        ]
        if include_output:
            columns.append("ao.output")

        sql = f"""
                SELECT {", ".join(columns)}
                FROM agent.agent_outputs ao
                LEFT JOIN agent.plan_runs pr
                  ON ao.plan_run_id = pr.plan_run_id
                LEFT JOIN agent.execution_plans ep
                  ON ao.plan_id = ep.plan_id
                WHERE {where_clause}
                ORDER BY created_at ASC;
                """
        rows = await self.pg.generic_read(sql, params)
        return rows

    async def get_agent_outputs_no_cache(
        self, agent_id: str, plan_run_id: Optional[str] = None
    ) -> List[AgentOutput]:

        start = time.perf_counter()
        rows = await self.get_agent_outputs_data_from_db(
            agent_id=agent_id, include_output=True, plan_run_id=plan_run_id
        )
        end = time.perf_counter()
        logger.info(
            f"Fetched outputs from DB for {agent_id=} {plan_run_id=} in {(end - start):.2f}s"
        )
        if not rows:
            return []

        start = end
        io_outputs = [
            load_io_type(row["output"]) if row["output"] else row["output"] for row in rows
        ]
        end = time.perf_counter()
        logger.info(f"`load_io_type` for {agent_id=} {plan_run_id=} in {(end - start):.2f}s")

        start = end
        tasks = [get_output_from_io_type(output, pg=self.pg) for output in io_outputs]
        output_values = await asyncio.gather(*tasks)
        logger.info(
            f"`get_output_from_io_type` for {agent_id=} {plan_run_id=} in {time.perf_counter() - start:.2f}s"
        )

        outputs: List[AgentOutput] = []
        for row, output_value in zip(rows, output_values):
            row["output"] = output_value
            row["shared"] = row["shared"] or False
            row["run_metadata"] = (
                RunMetadata.model_validate(row["run_metadata"]) if row["run_metadata"] else None
            )
            locked_tasks = row["locked_tasks"] if row["locked_tasks"] else set()
            row["is_locked"] = row["task_id"] in locked_tasks
            if row["plan"]:
                # Might be slightly inefficient, but in the scheme of things
                # probably not noticeable. We can revisit if needed. Need to do
                # this so that frontend knows which outputs depend on other
                # outputs in case of deleting.
                plan = ExecutionPlan.from_dict(row["plan"])
                node_dependency_map = plan.get_node_dependency_map()
                node_parent_map = plan.get_node_parent_map()
                for node, children in node_dependency_map.items():
                    if node.tool_task_id == row["task_id"]:
                        parents = node_parent_map.get(node, set())
                        row["dependent_task_ids"] = [node.tool_task_id for node in children]
                        # Only include parents if they are also outputs
                        row["parent_task_ids"] = [
                            node.tool_task_id for node in parents if node.is_output_node
                        ]

            output = AgentOutput(agent_id=agent_id, **row)
            outputs.append(output)

        return outputs

    async def write_tool_split_outputs(
        self, outputs_with_ids: List[OutputWithID], context: PlanRunContext
    ) -> None:
        now = get_now_utc()
        time_delta = datetime.timedelta(seconds=0.01)

        errmsg = "Failed to serialize the output for log"
        rows = [
            {
                "agent_id": context.agent_id,
                "plan_id": context.plan_id,
                "plan_run_id": context.plan_run_id,
                "log_id": output.output_id,
                "task_id": output.task_id,
                "log_data": safe_dump_io_type(output.output, errmsg),
                "is_task_output": True,
                "created_at": now + idx * time_delta,  # preserve order
            }
            for idx, output in enumerate(outputs_with_ids)
        ]
        await self.pg.multi_row_insert(table_name="agent.work_logs", rows=rows)

    @async_perf_logger
    async def get_agent_debug_tool_calls(self, agent_id: str) -> Dict[str, Any]:
        sql = """
        SELECT  plan_id, plan_run_id, task_id, tool_name, start_time_utc,
        end_time_utc , error_msg, replay_id, debug_info
        FROM agent.task_run_info
        WHERE agent_id = %(agent_id)s
        ORDER BY end_time_utc ASC
        """
        rows = await self.generic_read(sql, {"agent_id": agent_id})
        res: Dict[str, Any] = OrderedDict()
        tz = datetime.timezone.utc
        for row in rows:
            plan_run_id = row["plan_run_id"]
            if plan_run_id not in res:
                res[plan_run_id] = OrderedDict()
            tool_name = row["tool_name"]
            if row.get("start_time_utc"):
                row["start_time_utc"] = row["start_time_utc"].replace(tzinfo=tz).isoformat()
            if row.get("end_time_utc"):
                row["end_time_utc"] = row["end_time_utc"].replace(tzinfo=tz).isoformat()
            env_upper = EnvironmentUtils.aws_ssm_prefix.upper()
            row["replay_command"] = (
                f"ENVIRONMENT={env_upper} pipenv run python run_plan_task.py "
                f"--env {env_upper} --replay-id {row['replay_id']}"
            )
            row[f"args_{row['replay_id']}"] = {}
            row[f"result_{row['replay_id']}"] = {}
            if row["debug_info"]:
                row["debug_info"] = json.loads(row["debug_info"])
            res[plan_run_id][f"{tool_name}_{row['task_id']}"] = row
        return res

    async def get_agent_outputs_cache(
        self, cache: RedisCacheBackend, agent_id: str, plan_run_id: Optional[str] = None
    ) -> List[AgentOutput]:
        """
        if `plan_run_id` is None, get the latest run's outputs
        """
        start = time.perf_counter()
        # download metadata for all outputs (no values)
        rows_without_output = await self.get_agent_outputs_data_from_db(
            agent_id=agent_id, include_output=False, plan_run_id=plan_run_id
        )
        end = time.perf_counter()

        if not rows_without_output:
            rows_without_output = []

        logger.info(
            f"Total time to get output METADATA for {agent_id=} {plan_run_id=} from DB "
            f"is {(end - start):.2f} seconds, found {len(rows_without_output)} rows"
        )

        if not rows_without_output:
            return []

        start = end
        output_id_to_row = {row["output_id"]: row for row in rows_without_output}
        uncached_output_ids = set(output_id_to_row.keys())
        output_values: List[Tuple[str, Output]] = []

        outputs_with_ids: Optional[Dict[str, TextOutput | GraphOutput | TableOutput]] = (
            await cache.multiget(keys=output_id_to_row.keys())  # type: ignore
        )
        if outputs_with_ids:
            for cached_output_id, cached_output in outputs_with_ids.items():
                if cached_output:
                    uncached_output_ids.discard(cached_output_id)
                    output_values.append((cached_output_id, cached_output))

        end = time.perf_counter()
        logger.info(
            f"Total time to get output VALUES for {agent_id=} {plan_run_id=} from CACHE "
            f"is {(end - start):.2f} seconds, found {len(output_values)} rows"
        )

        logger.info(f"{len(uncached_output_ids)} uncached ids need to be fetched")

        if uncached_output_ids:
            start = end

            # meaning we need to download the output values from the database
            sql = """
                SELECT output_id::VARCHAR, output
                FROM agent.agent_outputs
                WHERE output_id = ANY(%(output_ids)s)
            """
            output_rows = await self.pg.generic_read(sql, {"output_ids": list(uncached_output_ids)})

            end = time.perf_counter()
            logger.info(
                f"Total time to get uncached output VALUES for {agent_id=} {plan_run_id=} from DB is "
                f"{(end - start):.2f} seconds, found {len(output_rows)} rows"
            )
            start = end

            non_cached_output_tasks = []
            for row in output_rows:
                output_id = row["output_id"]
                io_output = load_io_type(row["output"]) if row["output"] else row["output"]
                non_cached_output_tasks.append(get_output_from_io_type(io_output, pg=self.pg))

            end = time.perf_counter()
            logger.info(
                f"Total time to `load_io_type` for uncached output VALUES for {agent_id=} {plan_run_id=} is "
                f"{(end - start):.2f} seconds"
            )
            start = end

            results = await asyncio.gather(*non_cached_output_tasks)
            inputs_to_cache = {}
            for output_row, non_cached_output in zip(output_rows, results):
                output_id = output_row["output_id"]
                output_values.append((output_id, non_cached_output))
                inputs_to_cache[output_id] = non_cached_output

            # static analysis: ignore[missing_await]
            run_async_background(cache.multiset(inputs_to_cache, ttl=3600 * 24))  # type: ignore

            end = time.perf_counter()
            logger.info(
                "Total time to `get_output_from_io_type` for uncached output"
                f" VALUES for {agent_id=} {plan_run_id=} is {(end - start):.2f} seconds"
            )

        outputs: List[AgentOutput] = []
        for output_id, output_value in output_values:
            row = output_id_to_row[output_id]

            row["output"] = output_value
            row["shared"] = row["shared"] or False
            row["run_metadata"] = (
                RunMetadata.model_validate(row["run_metadata"]) if row["run_metadata"] else None
            )
            locked_tasks = row["locked_tasks"] if row["locked_tasks"] else set()
            row["is_locked"] = row["task_id"] in locked_tasks
            if row["plan"]:
                # Might be slightly inefficient, but in the scheme of things
                # probably not noticeable. We can revisit if needed. Need to do
                # this so that frontend knows which outputs depend on other
                # outputs in case of deleting.
                plan = ExecutionPlan.from_dict(row["plan"])
                node_dependency_map = plan.get_node_dependency_map()
                node_parent_map = plan.get_node_parent_map()
                for node, children in node_dependency_map.items():
                    if node.tool_task_id == row["task_id"]:
                        parents = node_parent_map.get(node, set())
                        row["dependent_task_ids"] = [node.tool_task_id for node in children]
                        # Only include parents if they are also outputs
                        row["parent_task_ids"] = [
                            node.tool_task_id for node in parents if node.is_output_node
                        ]

            output = AgentOutput(agent_id=agent_id, **row)
            outputs.append(output)

        outputs.sort(key=lambda x: x.created_at)
        return outputs

    async def delete_agent_outputs(self, agent_id: str, output_ids: List[str]) -> None:
        sql = """
        UPDATE agent.agent_outputs
        SET deleted = TRUE
        WHERE agent_id = %(agent_id)s AND output_id = ANY(%(output_ids)s)
        """
        await self.pg.generic_write(sql, {"agent_id": agent_id, "output_ids": output_ids})

    async def lock_plan_tasks(self, agent_id: str, plan_id: str, task_ids: List[str]) -> None:
        # Probably a faster/better way to do this, but should work for now
        sql = """
        SELECT locked_tasks FROM agent.execution_plans
        WHERE agent_id = %(agent_id)s AND plan_id = %(plan_id)s
        """
        rows = await self.pg.generic_read(
            sql, {"agent_id": agent_id, "plan_id": plan_id, "task_ids": task_ids}
        )
        if not rows:
            return
        locked_tasks = set(rows[0]["locked_tasks"])
        task_ids_set = locked_tasks.union(task_ids)

        sql = """
        UPDATE agent.execution_plans
        SET locked_tasks = %(task_ids)s
        WHERE agent_id = %(agent_id)s AND plan_id = %(plan_id)s
        """
        await self.pg.generic_write(
            sql, {"agent_id": agent_id, "plan_id": plan_id, "task_ids": list(task_ids_set)}
        )

    async def unlock_plan_tasks(self, agent_id: str, plan_id: str, task_ids: List[str]) -> None:
        sql = """
        UPDATE agent.execution_plans
        SET locked_tasks = (
          SELECT ARRAY(
            SELECT UNNEST(locked_tasks)
            EXCEPT
            SELECT UNNEST(%(task_ids)s::TEXT[])
          )
        )
        WHERE agent_id = %(agent_id)s AND plan_id = %(plan_id)s
        """
        await self.pg.generic_write(
            sql, {"agent_id": agent_id, "plan_id": plan_id, "task_ids": task_ids}
        )

    async def cancel_agent_plan(
        self, plan_id: Optional[str] = None, plan_run_id: Optional[str] = None
    ) -> None:
        cancelled_ids = [{"cancelled_id": _id} for _id in (plan_id, plan_run_id) if _id]
        if not cancelled_ids:
            return

        tasks = [self.pg.multi_row_insert(table_name="agent.cancelled_ids", rows=cancelled_ids)]
        if plan_run_id:
            sql = """
            UPDATE agent.plan_runs SET status = 'CANCELLED'
            WHERE plan_run_id = %(plan_run_id)s
            """
            tasks.append(self.pg.generic_write(sql, params={"plan_run_id": plan_run_id}))

        if plan_id:
            sql = """
            UPDATE agent.execution_plans SET status = 'CANCELLED'
            WHERE plan_id = %(plan_id)s
            """
            tasks.append(self.pg.generic_write(sql, params={"plan_id": plan_id}))

        await asyncio.gather(*tasks)

    async def is_cancelled(self, ids_to_check: List[str]) -> bool:
        """
        Returns true if ANY of the input ID's have been cancelled.
        """
        sql = """
        select * from agent.cancelled_ids where cancelled_id = ANY(%(ids_to_check)s)
        """
        rows = await self.pg.generic_read(sql, {"ids_to_check": ids_to_check})
        return len(rows) > 0

    async def get_cancelled_ids(self, ids_to_check: List[str]) -> List[str]:
        """
        Returns cancelled IDs from the given list.
        """
        sql = """
        SELECT cancelled_id FROM agent.cancelled_ids WHERE cancelled_id = ANY(%(ids_to_check)s)
        """
        rows = await self.pg.generic_read(sql, {"ids_to_check": ids_to_check})
        return [row["cancelled_id"] for row in rows]

    async def get_plan_run_outputs(self, plan_run_id: str) -> List[AgentOutput]:
        sql = """
                SELECT ao.agent_id::VARCHAR, ao.plan_id::VARCHAR, ao.plan_run_id::VARCHAR,
                  ao.output_id::VARCHAR, ao.task_id::VARCHAR, ao.is_intermediate,
                    ao.output, ao.created_at, COALESCE(pr.shared, FALSE) AS shared
                FROM agent.agent_outputs ao
                LEFT JOIN agent.plan_runs pr
                ON ao.plan_run_id = pr.plan_run_id
                WHERE ao.plan_run_id = %(plan_run_id)s
                  AND NOT ao.deleted
                ORDER BY created_at ASC;
                """
        rows = await self.pg.generic_read(sql, {"plan_run_id": plan_run_id})
        if not rows:
            return []

        outputs = []
        for row in rows:
            output = row["output"]
            output_value = load_io_type(output) if output else output
            output_value = await get_output_from_io_type(output_value, pg=self.pg)
            row["output"] = output_value
            row["shared"] = row["shared"] or False
            outputs.append(AgentOutput(**row))

        return outputs

    async def get_task_work_log_ids(
        self,
        agent_id: str,
        task_ids: List[str],
        plan_id: str,
    ) -> Dict[str, str]:
        """
        Returns a dict from task_id to log_id.
        """
        sql = """
        SELECT DISTINCT ON (task_id) log_id::TEXT, task_id::TEXT
        FROM agent.work_logs
        WHERE agent_id = %(agent_id)s AND plan_id = %(plan_id)s
              AND task_id = ANY(%(task_ids)s)
              AND is_task_output
        ORDER BY task_id, created_at DESC
        """
        rows = await self.pg.generic_read(
            sql, {"agent_id": agent_id, "task_ids": task_ids, "plan_id": plan_id}
        )
        result = {}
        for row in rows:
            result[row["task_id"]] = row["log_id"]
        return result

    async def get_task_outputs_from_work_log_ids(self, log_ids: List[str]) -> Dict[str, Any]:
        """
        Given a list of log_ids, return a mapping from task_id to log_data for each input ID.
        """
        sql = """
        SELECT task_id::TEXT, log_data FROM agent.work_logs
        WHERE log_id = ANY(%(log_ids)s)
            AND is_task_output AND log_data NOTNULL
        """
        rows = await self.pg.generic_read(sql, {"log_ids": log_ids})
        result = {}
        for row in rows:
            result[row["task_id"]] = load_io_type(row["log_data"])

        return result

    async def get_task_output(
        self, agent_id: str, plan_run_id: str, task_id: str
    ) -> Optional[IOType]:
        sql = """
        SELECT log_data
        FROM agent.work_logs
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s AND task_id = %(task_id)s
            AND is_task_output AND log_data NOTNULL AND viewable
        ORDER BY created_at DESC
        LIMIT 1;
        """
        rows = await self.pg.generic_read(
            sql, {"agent_id": agent_id, "plan_run_id": plan_run_id, "task_id": task_id}
        )
        if not rows:
            return None
        return load_io_type(rows[0]["log_data"])

    async def get_log_output(
        self, agent_id: str, plan_run_id: str, log_id: str
    ) -> Optional[IOType]:
        sql = """
        SELECT log_data
        FROM agent.work_logs
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s AND log_id = %(log_id)s
            AND NOT is_task_output AND log_data NOTNULL AND viewable
        ORDER BY created_at DESC
        LIMIT 1;
        """
        rows = await self.pg.generic_read(
            sql, {"agent_id": agent_id, "plan_run_id": plan_run_id, "log_id": log_id}
        )
        if not rows:
            return None
        return load_io_type(rows[0]["log_data"])

    @async_perf_logger
    async def get_latest_execution_plan(
        self, agent_id: str, only_finished_plans: bool = False, only_running_plans: bool = False
    ) -> Tuple[
        Optional[str],
        Optional[ExecutionPlan],
        Optional[datetime.datetime],
        Optional[str],
        Optional[str],
    ]:
        extra_where = ""
        if only_finished_plans:
            extra_where = "AND ep.status = 'READY'"
        elif only_running_plans:
            extra_where = "AND ep.status IN ('RUNNING', 'NOT_STARTED')"
        sql = f"""
            SELECT ep.plan_id::VARCHAR, ep.plan, COALESCE(pr.created_at, ep.created_at) AS created_at,
                ep.status, pr.plan_run_id::VARCHAR AS upcoming_plan_run_id, ep.locked_tasks
            FROM agent.execution_plans ep
            LEFT JOIN agent.plan_runs pr ON ep.plan_id = pr.plan_id
            WHERE ep.agent_id = %(agent_id)s {extra_where}
            ORDER BY ep.last_updated DESC, pr.created_at DESC
            LIMIT 1;
        """
        rows = await self.pg.generic_read(sql, params={"agent_id": agent_id})
        if not rows:
            return None, None, None, None, None
        plan = ExecutionPlan.from_dict(rows[0]["plan"])
        plan.locked_task_ids = list(rows[0]["locked_tasks"])
        return (
            rows[0]["plan_id"],
            plan,
            rows[0]["created_at"],
            rows[0]["status"],
            rows[0]["upcoming_plan_run_id"],
        )

    async def get_execution_plan_for_run(self, plan_run_id: str) -> Tuple[str, ExecutionPlan]:
        sql = """
            SELECT ep.plan_id::TEXT, ep.plan
            FROM agent.execution_plans ep
            JOIN agent.plan_runs pr ON ep.plan_id = pr.plan_id
            WHERE pr.plan_run_id = %(plan_run_id)s
        """
        rows = await self.pg.generic_read(sql, params={"plan_run_id": plan_run_id})
        return (rows[0]["plan_id"], ExecutionPlan.from_dict(rows[0]["plan"]))

    async def get_agent_owner(self, agent_id: str, include_deleted: bool = True) -> Optional[str]:
        """
        This function retrieves the owner of an agent, mainly used in authorization.
        Caches the result for 128 calls since the owner cannot change.

        Args:
            agent_id: The agent id to retrieve the owner for.

        Returns: The user id of the agent owner.
        """
        deleted_clause = "AND NOT deleted" if not include_deleted else ""
        sql = f"""
            SELECT user_id::VARCHAR
            FROM agent.agents
            WHERE agent_id = %(agent_id)s {deleted_clause};
        """
        rows = await self.pg.generic_read(sql, params={"agent_id": agent_id})
        return rows[0]["user_id"] if rows else None

    @async_perf_logger
    async def get_agent_plan_runs(
        self,
        agent_id: str,
        start_date: Optional[datetime.date] = None,  # inclusive
        end_date: Optional[datetime.date] = None,  # exclusive
        start_index: Optional[int] = 0,
        limit_num: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, str]], int]:
        params: Dict[str, Any] = {"agent_id": agent_id}

        start_date_filter = ""
        if start_date:
            params["start_date"] = start_date
            start_date_filter = " AND created_at >= %(start_date)s"

        end_date_filter = ""
        if end_date:
            params["end_date"] = end_date
            end_date_filter = " AND created_at < %(end_date)s"

        offset_sql = ""
        if start_index is not None and start_index > 0:
            offset_sql = "OFFSET %(start_index)s"
            params["start_index"] = start_index

        limit_sql = ""
        if limit_num is not None:
            limit_sql = "LIMIT %(limit_num)s"
            params["limit_num"] = limit_num

        sql = f"""
        SELECT plan_run_id::VARCHAR, plan_id::VARCHAR
        FROM agent.plan_runs
        WHERE agent_id = %(agent_id)s{start_date_filter}{end_date_filter}
        ORDER BY created_at DESC
        {limit_sql} {offset_sql}
        """

        total_count_sql = f"""
        SELECT COUNT(*) AS total_plan_count
        FROM agent.plan_runs
        WHERE agent_id = %(agent_id)s{start_date_filter}{end_date_filter}
        """

        rows, total_count_rows = await asyncio.gather(
            self.pg.generic_read(sql, params=params),
            self.pg.generic_read(total_count_sql, params=params),
        )

        total_plan_count = total_count_rows[0]["total_plan_count"] if total_count_rows else 0

        return [(row["plan_run_id"], row["plan_id"]) for row in rows], total_plan_count

    async def get_agent_name(self, agent_id: str) -> str:
        sql = """
        SELECT agent_name FROM agent.agents WHERE agent_id = %(agent_id)s LIMIT 1;
        """
        rows = await self.pg.generic_read(sql, params={"agent_id": agent_id})
        return rows[0]["agent_name"]

    async def get_existing_agents_names(self, user_id: str) -> List[str]:
        sql = """
        SELECT agent_name FROM agent.agents WHERE user_id = %(user_id)s;
        """
        rows = await self.pg.generic_read(sql, params={"user_id": user_id})
        return [row["agent_name"] for row in rows]

    @async_perf_logger
    async def get_agent_worklogs(
        self,
        agent_id: str,
        start_date: Optional[datetime.date] = None,  # inclusive
        end_date: Optional[datetime.date] = None,  # exclusive
        plan_run_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        params: Dict[str, Any] = {"agent_id": agent_id}
        filters = ""
        if start_date:
            filters += " AND wl.created_at >= %(start_date)s"
            params["start_date"] = start_date
        if end_date:
            filters += " AND wl.created_at < %(end_date)s"
            params["end_date"] = end_date
        if plan_run_ids:
            filters += " AND wl.plan_run_id = ANY(%(plan_run_ids)s)"
            params["plan_run_ids"] = plan_run_ids

        sql1 = f"""
            SELECT wl.plan_id::VARCHAR, wl.plan_run_id::VARCHAR, wl.task_id::VARCHAR, wl.is_task_output,
                wl.log_id::VARCHAR, wl.log_message, wl.created_at, pr.shared, (log_data NOTNULL) AS has_output,
                pr.run_metadata
            FROM agent.work_logs wl
            LEFT JOIN agent.plan_runs pr
            ON wl.plan_run_id = pr.plan_run_id
            AND wl.agent_id = pr.agent_id
            WHERE wl.agent_id = %(agent_id)s {filters}
            ORDER BY created_at DESC;
        """
        return await self.pg.generic_read(sql1, params=params)

    @async_perf_logger
    async def get_execution_plans(
        self, plan_ids: List[str]
    ) -> Dict[str, Tuple[ExecutionPlan, PlanStatus, datetime.datetime, datetime.datetime]]:
        sql = """
            SELECT plan_id::VARCHAR, plan, status, created_at, last_updated, locked_tasks
            FROM agent.execution_plans
            WHERE plan_id = ANY(%(plan_ids)s)
        """
        rows = await self.pg.generic_read(sql, params={"plan_ids": plan_ids})
        output = {}
        for row in rows:
            plan = ExecutionPlan.from_dict(row["plan"])
            plan.locked_task_ids = list(row["locked_tasks"])
            output[row["plan_id"]] = (
                plan,
                PlanStatus(row["status"]),
                row["created_at"],
                row["last_updated"],
            )

        return output

    async def get_chats_history_for_agent(
        self,
        agent_id: str,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        start_index: Optional[int] = 0,
        limit_num: Optional[int] = None,
    ) -> ChatContext:
        """
        Get chat history for an agent
        """
        params: Dict[str, Any] = {"agent_id": agent_id}

        dt_filter = ""
        if start:
            dt_filter += " AND cm.message_time >= %(start)s"
            params["start"] = start
        if end:
            dt_filter += " AND cm.message_time <= %(end)s"
            params["end"] = end

        offset_sql = ""
        if start_index is not None and start_index > 0:
            offset_sql = "OFFSET %(start_index)s"
            params["start_index"] = start_index

        limit_sql = ""
        if limit_num is not None:
            limit_sql = "LIMIT %(limit_num)s"
            params["limit_num"] = limit_num

        sql = f"""
            SELECT cm.message_id::VARCHAR, cm.message, cm.is_user_message, cm.message_time,
            cm.message_author, cm.plan_run_id::VARCHAR,
              COALESCE(nf.unread, FALSE) as unread, cm.message_metadata
            FROM agent.chat_messages cm
            LEFT JOIN agent.notifications nf
            ON cm.message_id = nf.message_id
            WHERE cm.agent_id = %(agent_id)s{dt_filter}
            ORDER BY cm.message_time DESC
            {limit_sql} {offset_sql}
        """
        total_count_sql = f"""
            SELECT COUNT(*) AS total_message_count
            FROM agent.chat_messages cm
            WHERE cm.agent_id = %(agent_id)s{dt_filter}
        """
        rows, total_count_rows = await asyncio.gather(
            self.pg.generic_read(sql, params=params),
            self.pg.generic_read(total_count_sql, params=params),
        )
        total_message_count = total_count_rows[0]["total_message_count"] if total_count_rows else 0

        return ChatContext(
            messages=[Message(agent_id=agent_id, **row) for row in rows],
            total_message_count=total_message_count,
        )

    async def create_agent(self, agent_metadata: AgentInfo) -> None:
        await self.pg.multi_row_insert(
            table_name="agent.agents", rows=[agent_metadata.to_agent_row()]
        )

    async def update_agent_draft_status(self, agent_id: str, is_draft: bool) -> None:
        sql = """
        UPDATE agent.agents SET is_draft = %(is_draft)s WHERE agent_id = %(agent_id)s
        """
        await self.pg.generic_write(sql, params={"agent_id": agent_id, "is_draft": is_draft})

    async def update_agent_help_requested(self, agent_id: str, help_requested: bool) -> None:
        sql = """
        UPDATE agent.agents SET help_requested = %(help_requested)s WHERE agent_id = %(agent_id)s
        """
        await self.pg.generic_write(
            sql, params={"agent_id": agent_id, "help_requested": help_requested}
        )

    async def update_execution_plan_status(
        self, plan_id: str, agent_id: str, status: PlanStatus = PlanStatus.READY
    ) -> None:

        now_utc = get_now_utc()
        sql = """
        INSERT INTO agent.execution_plans (plan_id, agent_id, plan, created_at, last_updated, status)
        VALUES (
          %(plan_id)s, %(agent_id)s, %(empty_plan)s, %(created_at)s, %(last_updated)s, %(status)s
        )
        ON CONFLICT (plan_id) DO UPDATE SET
          last_updated = %(now_utc)s,
          status = EXCLUDED.status
        """
        created_at = last_updated = get_now_utc()  # need so skip_commit db has proper times
        await self.pg.generic_write(
            sql,
            params={
                "plan_id": plan_id,
                "agent_id": agent_id,
                "created_at": created_at,
                # In case this is the first insert only
                "empty_plan": ExecutionPlan(nodes=[]).model_dump_json(),
                "last_updated": last_updated,
                "status": status.value,
                "now_utc": now_utc,
            },
        )

    async def write_execution_plan(
        self,
        plan_id: str,
        agent_id: str,
        plan: ExecutionPlan,
        status: PlanStatus = PlanStatus.READY,
    ) -> None:
        now_utc = get_now_utc()
        sql = """
        INSERT INTO agent.execution_plans (plan_id, agent_id, plan, created_at,
                    last_updated, status, locked_tasks)
        VALUES (
          %(plan_id)s, %(agent_id)s, %(plan)s, %(created_at)s, %(last_updated)s,
          %(status)s, %(locked_tasks)s
        )
        ON CONFLICT (plan_id) DO UPDATE SET
          agent_id = EXCLUDED.agent_id,
          plan = EXCLUDED.plan,
          last_updated = %(now_utc)s,
          status = EXCLUDED.status,
          locked_tasks = EXCLUDED.locked_tasks
        """
        created_at = last_updated = get_now_utc()  # need so skip_commit db has proper times
        locked_ids = plan.locked_task_ids
        await self.pg.generic_write(
            sql,
            params={
                "plan_id": plan_id,
                "agent_id": agent_id,
                "plan": plan.model_dump_json(),
                "created_at": created_at,
                "last_updated": last_updated,
                "status": status.value,
                "locked_tasks": locked_ids,
                "now_utc": now_utc,
            },
        )

    async def update_plan_run(
        self, agent_id: str, plan_id: str, plan_run_id: str, status: Status = Status.NOT_STARTED
    ) -> None:
        now_utc = get_now_utc()
        sql = """
        INSERT INTO agent.plan_runs (agent_id, plan_id, plan_run_id, status)
        VALUES (%(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(status)s)
        ON CONFLICT (plan_run_id) DO UPDATE SET
          agent_id = EXCLUDED.agent_id,
          plan_id = EXCLUDED.plan_id,
          status = EXCLUDED.status,
          last_updated = %(now_utc)s
        """
        await self.pg.generic_write(
            sql,
            params={
                "agent_id": agent_id,
                "plan_id": plan_id,
                "plan_run_id": plan_run_id,
                "status": status.value,
                "now_utc": now_utc,
            },
        )

    async def get_plan_run_statuses(self, plan_run_ids: List[str]) -> Dict[str, PlanRunStatusInfo]:
        sql = """
        SELECT plan_run_id::TEXT, created_at AS start_time, last_updated AS end_time, status
        FROM agent.plan_runs
        WHERE plan_run_id = ANY(%(plan_run_ids)s)
        """
        rows = await self.pg.generic_read(sql, {"plan_run_ids": plan_run_ids})
        return {row["plan_run_id"]: PlanRunStatusInfo(**row) for row in rows}

    async def get_task_run_info(
        self, plan_run_id: str, task_id: str, tool_name: str
    ) -> Optional[Tuple[str, str, str, datetime.datetime]]:
        """
        Given relevant ID's, return a tuple of:
        (args, result, debug info, timestamp)
        """
        sql = """
        SELECT tri.task_args, tri.debug_info, tri.created_at, tri.output
        FROM agent.task_run_info tri
        WHERE tri.task_id = %(task_id)s AND tri.plan_run_id = %(plan_run_id)s
          AND tri.tool_name = %(tool_name)s
        """
        rows = await self.pg.generic_read(
            sql, {"plan_run_id": plan_run_id, "task_id": task_id, "tool_name": tool_name}
        )
        if not rows:
            return None
        row = rows[0]
        return (row["task_args"], row["output"], row["debug_info"], row["created_at"])

    async def insert_task_run_info(
        self,
        context: PlanRunContext,
        tool_name: str,
        args: str,
        output: Optional[str],
        replay_id: str,
        start_time_utc: datetime.datetime,
        end_time_utc: datetime.datetime,
        error_msg: Optional[str] = "",
        debug_info: Optional[str] = None,
    ) -> None:
        sql = """
        INSERT INTO agent.task_run_info (task_id, agent_id, plan_run_id,
          tool_name, task_args, debug_info, output, error_msg, start_time_utc, end_time_utc, replay_id, plan_id,
          context)
        VALUES (%(task_id)s, %(agent_id)s, %(plan_run_id)s,
          %(tool_name)s, %(task_args)s, %(debug_info)s, %(output)s, %(error_msg)s,
          %(start_time_utc)s, %(end_time_utc)s, %(replay_id)s, %(plan_id)s, %(context)s)
        """
        if not error_msg:
            error_msg = ""
        await self.pg.generic_write(
            sql,
            {
                "task_id": context.task_id,
                "agent_id": context.agent_id,
                "plan_run_id": context.plan_run_id,
                "tool_name": tool_name,
                "task_args": args,
                "debug_info": debug_info,
                "output": output,
                "error_msg": error_msg,
                "end_time_utc": end_time_utc,
                "start_time_utc": start_time_utc,
                "replay_id": replay_id,
                "plan_id": context.plan_id,
                "context": context.model_dump_json(),
            },
        )

    async def get_task_outputs_from_replay_ids(self, replay_ids: List[str]) -> Dict[str, Any]:
        sql = """
            SELECT
                task_id::TEXT,
                output
            FROM agent.task_run_info
            WHERE
                replay_id = ANY(%(replay_ids)s)
                AND output IS NOT NULL
        """
        rows = await self.generic_read(sql, params={"replay_ids": replay_ids})
        res = {}
        for row in rows:
            res[row["task_id"]] = load_io_type(row["output"])
        return res

    async def get_task_run_statuses(
        self, plan_run_ids: List[str]
    ) -> Dict[Tuple[str, str], TaskRunStatusInfo]:
        """
        Returns a mapping from (plan_run_id, task_id) to task info
        """
        sql = """
        SELECT task_id::TEXT, plan_run_id::TEXT,
          created_at AS start_time, last_updated AS end_time, status
        FROM agent.task_runs
        WHERE plan_run_id = ANY(%(plan_run_ids)s)
        """
        rows = await self.pg.generic_read(sql, {"plan_run_ids": plan_run_ids})
        return {(row["plan_run_id"], row["task_id"]): TaskRunStatusInfo(**row) for row in rows}

    async def update_task_statuses(
        self,
        agent_id: str,
        tasks: List[TaskStatus],
        plan_run_id: str,
    ) -> None:
        now_utc = get_now_utc()
        sql = """
        INSERT INTO agent.task_runs AS tr (task_id, agent_id, plan_run_id, status)
        VALUES (%(task_id)s, %(agent_id)s, %(plan_run_id)s, %(status)s)
        ON CONFLICT (plan_run_id, task_id) DO UPDATE SET
          agent_id = EXCLUDED.agent_id,
          status = EXCLUDED.status,
          last_updated = %(now_utc)s WHERE tr.status != EXCLUDED.status
        """
        params = [
            {
                "agent_id": agent_id,
                "task_id": task.task_id,
                "plan_run_id": plan_run_id,
                "status": task.status.value,
                "now_utc": now_utc,
            }
            for task in tasks
        ]
        # TODO probably should have a better way of doing this...
        if isinstance(self.pg, AsyncPostgresBase):
            async with self.pg.cursor() as cursor:
                await cursor.executemany(sql, params)
        elif isinstance(self.pg, SyncBoostedPG):
            with self.pg.cursor() as cursor:
                cursor.executemany(sql, params)

    async def delete_agent_by_id(self, agent_id: str) -> None:
        await self.pg.generic_update(
            table_name="agent.agents",
            where={"agent_id": agent_id},
            values_to_update={"deleted": True},
        )

    async def is_agent_deleted(self, agent_id: Optional[str]) -> bool:
        if agent_id is None:
            return False

        sql = "SELECT deleted FROM agent.agents WHERE agent_id = %(agent_id)s"
        rows = await self.pg.generic_read(sql, {"agent_id": agent_id})
        if rows:
            return rows[0]["deleted"]
        return True

    async def is_agent_draft(self, agent_id: Optional[str]) -> bool:
        if agent_id is None:
            return False

        sql = "SELECT is_draft FROM agent.agents WHERE agent_id = %(agent_id)s"
        rows = await self.pg.generic_read(sql, {"agent_id": agent_id})
        if rows:
            return rows[0]["is_draft"]
        return True

    async def restore_agent_by_id(self, agent_id: str) -> None:
        await self.pg.generic_update(
            table_name="agent.agents",
            where={"agent_id": agent_id},
            values_to_update={"deleted": False},
        )

    async def get_agent_widget_title(self, output_id: str) -> str:
        sql = """
        SELECT output::JSONB->'title' AS widget_title
        FROM agent.agent_outputs
        WHERE output_id = %(output_id)s
        """
        rows = await self.pg.generic_read(sql, {"output_id": output_id})
        return rows[0]["widget_title"] if rows else ""

    async def update_agent_widget_name(
        self,
        agent_id: str,
        output_id: str,
        old_widget_title: str,
        new_widget_title: str,
        cache: RedisCacheBackend,
    ) -> None:

        rows = await self.get_agent_outputs_data_from_db(
            agent_id=agent_id, include_output=False, output_id=output_id
        )

        # atomically updating all the fields so that if one update were to fail,
        # the other updates don't get applied
        async with self.pg.transaction() as cursor:  # type: ignore
            if not rows:
                logger.warning(f"No rows found for agent_id: {agent_id}, output_id: {output_id}")
                return
            output = rows[0]

            await self.pg.generic_jsonb_update(
                table_name="agent.agent_outputs",
                jsonb_column="output",
                field_updates={"title": new_widget_title},
                where={"output_id": output_id},
                cursor=cursor,
            )

            run_summary_long = output["run_metadata"]["run_summary_long"]
            if run_summary_long:
                run_summary_long = Text.model_validate_json(json.dumps(run_summary_long))
                field_updates = {}

                # replace the old title with the new title in the run summary long by
                # replacing only the parent bullet point that matches `old_widget_title`
                old_run_summary_long: str = run_summary_long.val
                pattern = rf"(^- {re.escape(old_widget_title)}\s*$)"
                new_run_summary_long = re.sub(
                    pattern, rf"- {new_widget_title}", old_run_summary_long, flags=re.MULTILINE
                )
                field_updates.update({"run_summary_long.val": new_run_summary_long})

                # update the citation text offsets since the new widget title has different length
                citations = run_summary_long.history[0].citations
                if citations and old_widget_title in old_run_summary_long:
                    title_text_offset = len(new_widget_title) - len(old_widget_title)
                    widget_name_index = old_run_summary_long.index(old_widget_title)
                    new_citation_text_offsets = []
                    for citation in citations:
                        # only apply the offset to the citations that are after the widget title
                        if citation.citation_text_offset >= widget_name_index:
                            new_citation_text_offsets.append(
                                citation.citation_text_offset + title_text_offset
                            )
                        else:
                            new_citation_text_offsets.append(citation.citation_text_offset)
                    field_updates.update(
                        {
                            f"run_summary_long.history[0].citations[{i}].citation_text_offset": offset
                            for i, offset in enumerate(new_citation_text_offsets)
                        }
                    )

                await self.pg.generic_jsonb_update(
                    table_name="agent.plan_runs",
                    jsonb_column="run_metadata",
                    field_updates=field_updates,
                    where={"plan_run_id": output["plan_run_id"]},
                    cursor=cursor,
                )

            # update the title in execution plan so future plan runs will use the new title
            exeuction_plan = ExecutionPlan.model_validate(output["plan"])
            if exeuction_plan.nodes:
                i = 0
                for i, plan_node in enumerate(exeuction_plan.nodes):
                    if plan_node.is_output_node and plan_node.args["title"] == old_widget_title:
                        # setting i to the index of the node with the old title
                        break
                await self.pg.generic_jsonb_update(
                    table_name="agent.execution_plans",
                    jsonb_column="plan",
                    field_updates={f"nodes[{i}].args.title": new_widget_title},
                    where={"plan_id": output["plan_id"]},
                    cursor=cursor,
                )

        # invalidate the cache so we don't get the old title
        if cache:
            await cache.invalidate(output_id)

    async def get_user_agent_settings(self, user_id: str) -> AgentUserSettings:
        sql = """
        SELECT entity_type, include_web_results FROM agent.user_settings
        WHERE entity_id = %(user_id)s
        OR entity_id::TEXT IN
          (SELECT company_id FROM user_service.company_membership WHERE user_id = %(user_id)s::TEXT)
        LIMIT 2
        """
        rows = await self.pg.generic_read(sql, {"user_id": user_id})
        if not rows:
            return AgentUserSettings()
        if len(rows) == 1:
            row = rows[0]
            row.pop("entity_type")
            return AgentUserSettings(**row)
        entity_type_row_map = {}
        # Prioritize user over company settings
        for row in rows:
            entity_type = row.pop("entity_type")
            entity_type_row_map[entity_type] = row
        if AgentUserSettingsSource.USER in entity_type_row_map:
            return AgentUserSettings(**entity_type_row_map[AgentUserSettingsSource.USER])
        if AgentUserSettingsSource.COMPANY in entity_type_row_map:
            return AgentUserSettings(**entity_type_row_map[AgentUserSettingsSource.COMPANY])

        return AgentUserSettings()

    async def update_user_agent_settings(
        self,
        entity_id: str,
        entity_type: AgentUserSettingsSource,
        settings: AgentUserSettingsSetRequest,
    ) -> None:
        # Convert settings to a dictionary, filtering out None values
        settings_dict = {
            key: value for key, value in settings.model_dump().items() if value is not None
        }
        if not settings_dict:
            return

        columns = settings_dict.keys()
        values = [settings_dict[col] for col in columns]
        # Construct the ON CONFLICT update clause
        update_clause = SQL(", ").join(
            SQL("{} = EXCLUDED.{}").format(Identifier(col), Identifier(col)) for col in columns
        )

        query = SQL(
            """
            INSERT INTO agent.user_settings (entity_id, entity_type, {columns})
            VALUES (%s, %s, {placeholders})
            ON CONFLICT (entity_id) DO UPDATE
            SET {update_clause}
            """
        ).format(
            columns=SQL(", ").join(map(Identifier, columns)),
            placeholders=SQL(", ").join(Placeholder() for _ in columns),
            update_clause=update_clause,
        )

        await self.generic_write(query, [entity_id, entity_type] + values)  # type: ignore

    @async_perf_logger
    async def get_user_all_agents(
        self,
        user_id: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        include_deleted: bool = False,
    ) -> List[AgentInfo]:
        """
        This function retrieves all agents for a given user, optionally filtered
        by a list of agent ids.

        Args:
            user_id: The user id to retrieve agents for.
            agent_ids: The list of agent id's to filter by

        Returns: A list of all agents for the user, optionally filtered.
        """
        if not include_deleted:
            agent_where_clauses = ["not deleted"]
        else:
            agent_where_clauses = []

        params: Dict[str, Any] = {}
        if user_id:
            params["user_id"] = user_id
            agent_where_clauses.append("a.user_id = %(user_id)s")
        if agent_ids:
            params["agent_ids"] = agent_ids
            agent_where_clauses.append("a.agent_id = ANY(%(agent_ids)s)")

        agent_where_clause = ""
        if agent_where_clauses:
            agent_where_clause = "WHERE " + " AND ".join(agent_where_clauses)

        sql = f"""
        WITH a_id AS (
            SELECT agent_id, user_id, agent_name, created_at, last_updated,
                   automation_enabled, schedule, section_id, deleted, is_draft,
                   help_requested
            FROM agent.agents a
            {agent_where_clause}
        )
        SELECT
            a.agent_id::VARCHAR, a.user_id::VARCHAR, a.agent_name, a.created_at, a.last_updated,
            a.automation_enabled, a.section_id::VARCHAR, lr.last_run, msg.latest_agent_message,
            nu.num_unread AS unread_notification_count, a.schedule, lr.run_metadata,
            a.deleted, a.is_draft, lo.output_last_updated, a.help_requested
        FROM a_id AS a
        LEFT JOIN LATERAL (
            SELECT created_at AS last_run, run_metadata
            FROM agent.plan_runs AS pr
            WHERE pr.agent_id = a.agent_id
            ORDER BY pr.created_at DESC
            LIMIT 1
        ) AS lr ON true
        LEFT JOIN LATERAL (
            SELECT message AS latest_agent_message
            FROM agent.chat_messages AS m
            WHERE m.agent_id = a.agent_id
            ORDER BY m.message_time DESC
            LIMIT 1
        ) AS msg ON true
        LEFT JOIN (
            SELECT agent_id, COUNT(*) AS num_unread
            FROM agent.notifications
            WHERE unread
            GROUP BY agent_id
        ) AS nu ON nu.agent_id = a.agent_id
        LEFT JOIN LATERAL (
            SELECT created_at AS output_last_updated
            FROM agent.agent_outputs AS ao
            WHERE ao.agent_id = a.agent_id AND NOT ao.deleted
            ORDER BY ao.created_at DESC
            LIMIT 1
        ) AS lo ON true;
        """
        rows = await self.pg.generic_read(sql, params=params)
        output = []
        for row in rows:
            schedule = AgentSchedule.model_validate(row["schedule"]) if row["schedule"] else None
            # Handle this for backwards compatibility
            if row["automation_enabled"] and schedule is None:
                schedule = AgentSchedule.default()
            run_metadata = (
                RunMetadata.model_validate(row["run_metadata"]) if row["run_metadata"] else None
            )
            notification_string = (
                run_metadata.run_summary_short
                if run_metadata and run_metadata.run_summary_short
                else row["latest_agent_message"]
            )
            output.append(
                AgentInfo(
                    agent_id=row["agent_id"],
                    user_id=row["user_id"],
                    agent_name=row["agent_name"],
                    created_at=row["created_at"],
                    is_draft=row["is_draft"],
                    last_updated=row["last_updated"],
                    last_run=row["last_run"],
                    next_run=(schedule.get_next_run() if schedule else None),
                    latest_notification_string=notification_string,
                    automation_enabled=row["automation_enabled"],
                    unread_notification_count=row["unread_notification_count"] or 0,
                    schedule=schedule,
                    section_id=row["section_id"],
                    deleted=row["deleted"],
                    output_last_updated=row["output_last_updated"],
                    help_requested=row["help_requested"],
                )
            )
        return output

    async def update_agent_name(self, agent_id: str, agent_name: str) -> None:
        return await self.pg.generic_update(
            table_name="agent.agents",
            where={"agent_id": agent_id},
            values_to_update={"agent_name": agent_name, "last_updated": get_now_utc()},
        )

    async def insert_chat_messages(self, messages: List[Message]) -> None:
        await self.pg.multi_row_insert(
            table_name="agent.chat_messages", rows=[msg.to_message_row() for msg in messages]
        )

    async def insert_notifications(self, notifications: List[Notification]) -> None:
        await self.pg.multi_row_insert(
            table_name="agent.notifications", rows=[notif.model_dump() for notif in notifications]
        )

    async def get_notification_event_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        sql = """
        SELECT summary AS latest_notification_string,
        (
            SELECT COUNT(unread) FROM agent.notifications
            WHERE agent_id = %(agent_id)s AND unread = TRUE
        ) AS unread_count
        FROM agent.notifications
        WHERE agent_id = %(agent_id)s
        ORDER BY created_at DESC LIMIT 1;
        """

        rows = await self.pg.generic_read(sql, params={"agent_id": agent_id})
        if not rows:
            return None

        return rows[0]

    async def set_plan_run_share_status(self, plan_run_id: str, status: bool) -> None:
        await self.pg.generic_update(
            table_name="agent.plan_runs",
            where={"plan_run_id": plan_run_id},
            values_to_update={"shared": status},
        )

    async def mark_notifications_as_read(
        self, agent_id: str, timestamp: Optional[datetime.datetime] = None
    ) -> None:
        where_clause = "agent_id = %(agent_id)s"
        if timestamp is not None:
            where_clause += " AND created_at <= %(timestamp)s"

        sql = f"""
        UPDATE agent.notifications SET unread = FALSE WHERE {where_clause}
        """

        params = {"agent_id": agent_id, "timestamp": timestamp}

        await self.pg.generic_write(sql, params=params)

    async def mark_notifications_as_unread(self, agent_id: str, message_id: str) -> None:
        sql = """
        SELECT message_time FROM agent.chat_messages
        WHERE agent_id = %(agent_id)s AND message_id = %(message_id)s
        LIMIT 1
        """
        rows = await self.pg.generic_read(sql, {"agent_id": agent_id, "message_id": message_id})
        if not rows:
            return None

        message_timestamp = rows[0]["message_time"]
        sql = """
        UPDATE agent.notifications
        SET unread = TRUE
        WHERE agent_id = %(agent_id)s AND created_at >= %(timestamp)s
          AND message_id IS NOT NULL
        """
        await self.pg.generic_write(sql, {"agent_id": agent_id, "timestamp": message_timestamp})

    async def set_agent_automation_enabled(self, agent_id: str, enabled: bool) -> None:
        await self.pg.generic_update(
            table_name="agent.agents",
            where={"agent_id": agent_id},
            values_to_update={"automation_enabled": enabled},
        )

    async def get_agent_automation_enabled(self, agent_id: str) -> bool:
        sql = "SELECT automation_enabled from agent.agents where agent_id = %(agent_id)s"
        rows = await self.pg.generic_read(sql, {"agent_id": agent_id})
        return rows[0]["automation_enabled"]

    async def set_latest_plan_for_automated_run(self, agent_id: str) -> None:
        set_automated_run_plan_sql = """
                WITH latest_plan AS (
                    SELECT plan_id from agent.execution_plans where agent_id = %(agent_id)s AND status = 'READY'
                    ORDER BY created_at DESC LIMIT 1
                )
                UPDATE agent.execution_plans
                SET automated_run = TRUE
                WHERE plan_id IN (SELECT plan_id from latest_plan)
                """
        await self.pg.generic_write(set_automated_run_plan_sql, {"agent_id": agent_id})

        set_live_plan_outputs_sql = """
        WITH latest_plan AS (
                SELECT plan_id from agent.execution_plans where agent_id = %(agent_id)s AND status = 'READY'
                ORDER BY created_at DESC LIMIT 1
        )
        UPDATE agent.agent_outputs
        SET live_plan_output = TRUE
        WHERE plan_id in (SELECT plan_id from latest_plan)
        """
        await self.pg.generic_write(set_live_plan_outputs_sql, {"agent_id": agent_id})

    async def update_agent_schedule(self, agent_id: str, schedule: AgentSchedule) -> None:
        await self.pg.generic_update(
            table_name="agent.agents",
            where={"agent_id": agent_id},
            values_to_update={"schedule": schedule.model_dump_json()},
        )

    async def get_agent_schedule(self, agent_id: str) -> Optional[AgentSchedule]:
        sql = """
        SELECT schedule FROM agent.agents
        WHERE agent_id = %(agent_id)s
        """
        rows = await self.pg.generic_read(sql, params={"agent_id": agent_id})
        if not rows:
            return None
        schedule = rows[0]["schedule"]
        return AgentSchedule.model_validate(schedule) if schedule else None

    async def insert_agent_custom_notification(self, cn: CustomNotification) -> None:
        """
        Inserts a custom notification prompt for an agent, and returns the
        custom_notification_id.
        """
        sql = """
        INSERT INTO agent.custom_notifications
          (custom_notification_id, agent_id, notification_prompt, created_at, auto_generated)
        VALUES (%(custom_notification_id)s, %(agent_id)s,
               %(prompt)s, %(created_at)s, %(auto_generated)s)
        """
        # need to manually insert created_at for offline tool
        await self.pg.generic_write(
            sql,
            {
                "custom_notification_id": cn.custom_notification_id,
                "agent_id": cn.agent_id,
                "prompt": cn.notification_prompt,
                "created_at": cn.created_at,
                "auto_generated": cn.auto_generated,
            },
        )

    async def delete_agent_custom_notification_prompt(
        self, agent_id: str, custom_notification_id: str
    ) -> None:
        sql = """
        DELETE FROM agent.custom_notifications
        WHERE agent_id=%(agent_id)s
          AND custom_notification_id = %(custom_notification_id)s
        """
        await self.pg.generic_write(
            sql, params={"agent_id": agent_id, "custom_notification_id": custom_notification_id}
        )

    async def get_all_agent_custom_notifications(self, agent_id: str) -> List[CustomNotification]:
        sql = """
        SELECT custom_notification_id::TEXT, agent_id::TEXT,
          notification_prompt, created_at, auto_generated
        FROM agent.custom_notifications
        WHERE agent_id=%(agent_id)s
        ORDER BY created_at DESC
        """
        rows = await self.pg.generic_read(sql, {"agent_id": agent_id})
        return [CustomNotification(**row) for row in rows]

    async def set_plan_run_metadata(self, context: PlanRunContext, metadata: RunMetadata) -> None:
        sql = """
        UPDATE agent.plan_runs
        SET run_metadata = %(metadata)s
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s
        """
        await self.pg.generic_write(
            sql=sql,
            params={
                "metadata": dump_io_type(metadata),
                "agent_id": context.agent_id,
                "plan_run_id": context.plan_run_id,
            },
        )

    async def set_agent_subscriptions(
        self,
        agent_id: str,
        email_to_user_id: Dict[str, str],
        delete_previous_emails: bool = True,
    ) -> None:
        records_to_upload = []
        already_seen = set()
        # remove all the subs before adding new ones to avoid
        # duplicates, when calling from enable_agent_automation
        # need to make sure we do not remove all the emails
        if delete_previous_emails:
            await self.delete_all_email_subscriptions_for_agent(agent_id)
        for email, user_id in email_to_user_id.items():
            if email not in already_seen:
                already_seen.add(email)
                row = {
                    "agent_id": agent_id,
                    "email": email,
                    "user_id": user_id,
                }
                records_to_upload.append(row)
        await self.pg.multi_row_insert(
            table_name="agent.agent_notifications", rows=records_to_upload
        )

    async def delete_agent_emails(self, agent_id: str, email: str) -> None:
        sql = """
        DELETE FROM agent.agent_notifications
        WHERE agent_id=%(agent_id)s
          AND email = %(email)s
        """
        await self.pg.generic_write(sql, params={"agent_id": agent_id, "email": email})

    async def delete_all_email_subscriptions_for_agent(self, agent_id: str) -> None:
        sql = """
        DELETE FROM agent.agent_notifications
        WHERE agent_id=%(agent_id)s
        """
        await self.pg.generic_write(sql, params={"agent_id": agent_id})

    async def get_agent_subscriptions(self, agent_id: str) -> List[AgentNotificationEmail]:
        sql = """
        SELECT agent_id::TEXT, user_id, email
        FROM agent.agent_notifications
        WHERE agent_id=%(agent_id)s
        """
        rows = await self.pg.generic_read(sql, {"agent_id": agent_id})
        return [AgentNotificationEmail(**row) for row in rows]

    async def set_agent_section(
        self, new_section_id: Optional[str], agent_id: str, user_id: str
    ) -> None:
        sql = """
        UPDATE agent.agents SET section_id = %(new_section_id)s
        WHERE agent_id = %(agent_id)s and user_id = %(user_id)s
        """
        await self.pg.generic_write(
            sql=sql,
            params={"new_section_id": new_section_id, "agent_id": agent_id, "user_id": user_id},
        )

    async def update_agent_sections(
        self, new_section_id: Optional[str], section_id: str, user_id: str
    ) -> None:
        sql = """
        UPDATE agent.agents SET section_id = %(new_section_id)s
        WHERE section_id = %(section_id)s and user_id = %(user_id)s
        """
        await self.pg.generic_write(
            sql=sql,
            params={"new_section_id": new_section_id, "section_id": section_id, "user_id": user_id},
        )

    async def get_sidebar_sections(self, user_id: str) -> List[SidebarSection]:
        sql = """
        SELECT sections from agent.sidebar_sections where user_id = %(user_id)s
        """
        rows = await self.pg.generic_read(sql=sql, params={"user_id": user_id})
        if not rows:
            return []
        return [SidebarSection(**section) for section in rows[0]["sections"]]

    async def set_sidebar_sections(self, user_id: str, sections: List[SidebarSection]) -> None:
        sql = """
        INSERT INTO agent.sidebar_sections (user_id, sections) VALUES (%(user_id)s, %(sections)s)
                ON CONFLICT (user_id)
                DO UPDATE SET sections = EXCLUDED.sections
        """
        await self.pg.generic_write(
            sql,
            params={
                "user_id": user_id,
                "sections": json.dumps(
                    [section.model_dump(exclude_none=True) for section in sections]
                ),
            },
        )

    async def insert_quick_thought_for_agent(
        self, quick_thought_id: str, agent_id: str, quick_thoughts: QuickThoughts
    ) -> None:
        sql = """
        INSERT INTO agent.quick_thoughts (quick_thought_id, agent_id, quick_thought_text)
        VALUES (%(quick_thought_id)s, %(agent_id)s, %(quick_thought)s)
        """
        await self.pg.generic_write(
            sql,
            params={
                "agent_id": agent_id,
                "quick_thought": quick_thoughts.summary.model_dump_json(),
                "quick_thought_id": quick_thought_id,
            },
        )

    async def get_latest_quick_thought_for_agent(self, agent_id: str) -> Optional[QuickThoughts]:
        sql = """
        SELECT quick_thought_text FROM agent.quick_thoughts
        WHERE agent_id = %(agent_id)s
        ORDER BY created_at DESC
        LIMIT 1
        """
        rows = await self.pg.generic_read(
            sql,
            params={
                "agent_id": agent_id,
            },
        )
        if not rows:
            return None
        return QuickThoughts(summary=TextOutput.model_validate_json(rows[0]["quick_thought_text"]))

    async def set_agent_feedback(
        self, feedback_data: SetAgentFeedBackRequest, user_id: str
    ) -> None:
        # Prepare the parameters
        params = {
            "agent_id": feedback_data.agent_id,
            "plan_id": feedback_data.plan_id,
            "plan_run_id": feedback_data.plan_run_id,
            "output_id": feedback_data.output_id,
            "widget_title": feedback_data.widget_title,
            "rating": feedback_data.rating,
            "feedback_comment": feedback_data.feedback_comment,
            "feedback_user_id": user_id,
        }

        # Do not overwrite comment if None
        feedback_comment_clause = ""
        if feedback_data.feedback_comment is not None:
            feedback_comment_clause = "feedback_comment = EXCLUDED.feedback_comment,"

        sql = f"""
        INSERT INTO agent.feedback (
            agent_id, plan_id, plan_run_id, output_id, widget_title, rating, feedback_comment, feedback_user_id
        )
        VALUES (
            %(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(output_id)s, %(widget_title)s, %(rating)s,
            %(feedback_comment)s, %(feedback_user_id)s
        )
        ON CONFLICT (agent_id, plan_id, plan_run_id, output_id, feedback_user_id) DO UPDATE SET
          last_updated = NOW(),
          {feedback_comment_clause}
          rating = EXCLUDED.rating
        """

        await self.pg.generic_write(
            sql,
            params=params,
        )

    async def get_agent_feedback(
        self, agent_id: str, plan_id: str, plan_run_id: str, output_id: str, feedback_user_id: str
    ) -> List[AgentFeedback]:
        sql = """
            SELECT
                agent_id::TEXT,
                plan_id::TEXT,
                plan_run_id::TEXT,
                output_id,
                created_at,
                last_updated,
                widget_title,
                rating,
                feedback_comment,
                feedback_user_id
            FROM
                agent.feedback
            WHERE
                agent_id = %(agent_id)s
                AND plan_id = %(plan_id)s
                AND plan_run_id = %(plan_run_id)s
                AND output_id = %(output_id)s
                AND feedback_user_id = %(feedback_user_id)s
            """
        rows = await self.pg.generic_read(
            sql,
            {
                "agent_id": agent_id,
                "plan_id": plan_id,
                "plan_run_id": plan_run_id,
                "output_id": output_id,
                "feedback_user_id": feedback_user_id,
            },
        )
        return [AgentFeedback(**row) for row in rows]

    async def update_prompt_template(self, prompt_template: PromptTemplate) -> None:
        sql = """
        UPDATE agent.prompt_templates
        SET created_at = %(created_at)s,
            name = %(name)s, description = %(description)s, prompt = %(prompt)s,
            category = %(category)s, plan = %(plan)s, organization_ids = %(organization_ids)s,
            recommended_company_ids = %(recommended_company_ids)s, cadence_tag = %(cadence_tag)s,
            notification_criteria = %(notification_criteria)s

        WHERE template_id = %(template_id)s
        """
        await self.pg.generic_write(
            sql,
            {
                "created_at": prompt_template.created_at,
                "name": prompt_template.name,
                "description": prompt_template.description,
                "prompt": prompt_template.prompt,
                "category": prompt_template.category,
                "plan": prompt_template.plan.model_dump_json(),
                "cadence_tag": prompt_template.cadence_tag,
                "notification_criteria": prompt_template.notification_criteria,
                "organization_ids": prompt_template.organization_ids,
                "recommended_company_ids": prompt_template.recommended_company_ids,
                "template_id": prompt_template.template_id,
            },
        )

    async def insert_prompt_template(self, prompt_template: PromptTemplate) -> None:
        sql = """
        INSERT INTO agent.prompt_templates
          (template_id, name, description, prompt,
          category, created_at, plan, cadence_tag, notification_criteria,
          organization_ids, recommended_company_ids, user_id, description_embedding)
        VALUES (%(template_id)s, %(name)s, %(description)s, %(prompt)s,
                %(category)s, %(created_at)s, %(plan)s, %(cadence_tag)s, %(notification_criteria)s,
                %(organization_ids)s, %(recommended_company_ids)s,
                %(user_id)s, %(description_embedding)s)
        """
        await self.pg.generic_write(
            sql,
            {
                "template_id": prompt_template.template_id,
                "name": prompt_template.name,
                "description": prompt_template.description,
                "prompt": prompt_template.prompt,
                "category": prompt_template.category,
                "created_at": prompt_template.created_at,
                "plan": prompt_template.plan.model_dump_json(),
                "cadence_tag": prompt_template.cadence_tag,
                "notification_criteria": prompt_template.notification_criteria,
                "organization_ids": prompt_template.organization_ids,
                "recommended_company_ids": prompt_template.recommended_company_ids,
                "user_id": prompt_template.user_id,
                "description_embedding": prompt_template.description_embedding,
            },
        )

    async def get_prompt_templates_matched_query(
        self, query_embedding: List[float]
    ) -> List[PromptTemplate]:
        sql = """
        SELECT template_id::TEXT, name::TEXT, description::TEXT, prompt::TEXT, category, created_at,
            plan, cadence_tag::TEXT, notification_criteria, organization_ids, user_id, description_embedding,
            recommended_company_ids,
            1 - (%(query_embedding)s::VECTOR <=> description_embedding) AS similarity
        FROM agent.prompt_templates
        ORDER BY similarity DESC
        LIMIT 10
        """
        rows = await self.pg.generic_read(sql=sql, params={"query_embedding": query_embedding})
        if not rows:
            return []
        res = []
        for row in rows:
            res.append(
                PromptTemplate(
                    template_id=row["template_id"],
                    name=row["name"],
                    description=row["description"],
                    prompt=row["prompt"],
                    category=row["category"],
                    created_at=row["created_at"],
                    organization_ids=(
                        [str(company_id) for company_id in row["organization_ids"]]
                        if row["organization_ids"]
                        else None
                    ),
                    plan=ExecutionPlan.model_validate(row["plan"]),
                    cadence_tag=str(row["cadence_tag"]),
                    notification_criteria=(
                        [str(row) for row in row["notification_criteria"]]
                        if row["notification_criteria"]
                        else None
                    ),
                    user_id=str(row["user_id"]) if row["user_id"] else None,
                    recommended_company_ids=(
                        [str(company_id) for company_id in row["recommended_company_ids"]]
                        if row["recommended_company_ids"]
                        else None
                    ),
                )
            )

        return res

    async def get_prompt_templates(self) -> List[PromptTemplate]:
        sql = """
        SELECT template_id::TEXT, name::TEXT, description::TEXT, prompt::TEXT, category, created_at,
            plan, cadence_tag::TEXT, notification_criteria, organization_ids, user_id, recommended_company_ids
        FROM agent.prompt_templates
        ORDER BY name ASC
        """
        rows = await self.pg.generic_read(sql=sql)
        if not rows:
            return []
        res = []
        for row in rows:
            res.append(
                PromptTemplate(
                    template_id=row["template_id"],
                    name=row["name"],
                    description=row["description"],
                    prompt=row["prompt"],
                    category=row["category"],
                    created_at=row["created_at"],
                    organization_ids=(
                        [str(company_id) for company_id in row["organization_ids"]]
                        if row["organization_ids"]
                        else None
                    ),
                    plan=ExecutionPlan.model_validate(row["plan"]),
                    cadence_tag=str(row["cadence_tag"]),
                    notification_criteria=(
                        [str(row) for row in row["notification_criteria"]]
                        if row["notification_criteria"]
                        else None
                    ),
                    user_id=str(row["user_id"]) if row["user_id"] else None,
                    recommended_company_ids=(
                        [str(company_id) for company_id in row["recommended_company_ids"]]
                        if row["recommended_company_ids"]
                        else None
                    ),
                )
            )

        return res

    async def get_all_companies(self) -> List[UserOrganization]:

        sql = """
        SELECT id, name FROM user_service.organizations
        """
        rows = await self.pg.generic_read(sql=sql)
        return [
            UserOrganization(
                organization_id=str(row["id"]),
                organization_name=row["name"],
            )
            for row in rows
        ]

    async def delete_prompt_template(self, template_id: str) -> None:
        sql = """
        DELETE FROM agent.prompt_templates
        WHERE template_id = %(template_id)s
        """
        await self.pg.generic_write(sql, params={"template_id": template_id})

    async def get_org_name(self, org_id: str) -> str:
        sql = """
        select name from user_service.organizations
        where id = %(org_id)s
        """
        rows = await self.pg.generic_read(sql=sql, params={"org_id": org_id})
        return rows[0]["name"]

    async def copy_agent(
        self,
        src_agent_id: str,
        dst_agent_id: str,
        dst_user_id: str,
        dst_agent_name: Optional[str] = None,
    ) -> None:
        get_agent_sql = """
        select * from agent.agents where agent_id = %(agent_id)s
        """
        agent_info = (await self.pg.generic_read(get_agent_sql, {"agent_id": src_agent_id}))[0]
        agent_info["agent_id"] = dst_agent_id
        agent_info["user_id"] = dst_user_id
        if dst_agent_name is not None:
            agent_info["agent_name"] = dst_agent_name
        agent_info["created_at"] = datetime.datetime.utcnow()
        agent_info["last_updated"] = datetime.datetime.utcnow()
        meta = (
            AgentMetadata.model_validate(agent_info["agent_metadata"])
            if agent_info.get("agent_metadata")
            else AgentMetadata()
        )
        meta.copied_from_agent_id = src_agent_id
        agent_info["agent_metadata"] = meta.model_dump_json()
        agent_info["section_id"] = None
        if agent_info["schedule"]:
            agent_info["schedule"] = json.dumps(agent_info["schedule"])
        to_insert = []
        to_insert.append(InsertToTableArgs(table_name="agent.agents", rows=[agent_info]))

        execution_plan_sql = """
        select * from agent.execution_plans ep where agent_id = %(agent_id)s
        """
        execution_plans = await self.pg.generic_read(
            execution_plan_sql, params={"agent_id": src_agent_id}
        )
        execution_plan_id_map = {}
        for execution_plan in execution_plans:
            new_plan_id = str(uuid.uuid4())
            old_plan_id = execution_plan["plan_id"]
            execution_plan_id_map[old_plan_id] = new_plan_id
            execution_plan["plan_id"] = new_plan_id
            execution_plan["agent_id"] = dst_agent_id
            execution_plan["plan"] = json.dumps(execution_plan["plan"])

        to_insert.append(
            InsertToTableArgs(table_name="agent.execution_plans", rows=execution_plans)
        )

        plan_run_sql = """
                        select * from agent.plan_runs where agent_id = %(agent_id)s
                        """
        plan_runs = await self.pg.generic_read(plan_run_sql, params={"agent_id": src_agent_id})
        plan_run_id_map = {}
        for plan_run in plan_runs:
            new_plan_run_id = str(uuid.uuid4())
            old_plan_run_id = plan_run["plan_run_id"]
            plan_run["plan_id"] = execution_plan_id_map[plan_run["plan_id"]]
            plan_run_id_map[old_plan_run_id] = new_plan_run_id
            plan_run["plan_run_id"] = new_plan_run_id
            plan_run["agent_id"] = dst_agent_id
            plan_run["run_metadata"] = json.dumps(plan_run["run_metadata"])

        to_insert.append(InsertToTableArgs(table_name="agent.plan_runs", rows=plan_runs))

        get_chat_messages_sql = """
            select * from agent.chat_messages where agent_id = %(agent_id)s
        """
        chat_messages = await self.pg.generic_read(
            get_chat_messages_sql, {"agent_id": src_agent_id}
        )
        null_plan_run_id_messages = []
        regular_chat_messages = []
        for chat_message_data in chat_messages:
            chat_message_data["agent_id"] = dst_agent_id
            chat_message_data["message_id"] = str(uuid.uuid4())
            if chat_message_data["message_metadata"]:
                chat_message_data["message_metadata"] = json.dumps(
                    chat_message_data["message_metadata"]
                )
            else:
                chat_message_data["message_metadata"] = "{}"
            if chat_message_data["plan_run_id"]:
                regular_chat_messages.append(chat_message_data)
            else:
                null_plan_run_id_messages.append(chat_message_data)

        if regular_chat_messages:
            to_insert.append(
                InsertToTableArgs(table_name="agent.chat_messages", rows=regular_chat_messages)
            )
        if null_plan_run_id_messages:
            to_insert.append(
                InsertToTableArgs(table_name="agent.chat_messages", rows=null_plan_run_id_messages)
            )

        worklog_sql = """
        select * from agent.work_logs where agent_id = %(agent_id)s
        """

        work_log_entries = await self.pg.generic_read(worklog_sql, {"agent_id": src_agent_id})
        null_log_data_entries = []
        null_log_message_entries = []
        regular_entries = []
        for work_log_entry in work_log_entries:
            work_log_entry["agent_id"] = dst_agent_id
            work_log_entry["log_id"] = str(uuid.uuid4())
            work_log_entry["plan_id"] = execution_plan_id_map[work_log_entry["plan_id"]]
            work_log_entry["plan_run_id"] = plan_run_id_map[work_log_entry["plan_run_id"]]
            if not work_log_entry["log_data"]:
                null_log_data_entries.append(work_log_entry)
            elif not work_log_entry["log_message"]:
                null_log_message_entries.append(work_log_entry)
            else:
                regular_entries.append(work_log_entry)
        if regular_entries:
            to_insert.append(InsertToTableArgs(table_name="agent.work_logs", rows=regular_entries))
        if null_log_data_entries:
            to_insert.append(
                InsertToTableArgs(table_name="agent.work_logs", rows=null_log_data_entries)
            )
        if null_log_message_entries:
            to_insert.append(
                InsertToTableArgs(table_name="agent.work_logs", rows=null_log_message_entries)
            )

        outputs_sql = """select * from agent.agent_outputs where agent_id = %(agent_id)s"""
        agent_outputs = await self.pg.generic_read(outputs_sql, {"agent_id": src_agent_id})
        regular_output_entries = []
        null_task_id_output_entries = []
        for agent_output in agent_outputs:
            agent_output["output_id"] = str(uuid.uuid4())
            agent_output["agent_id"] = dst_agent_id
            agent_output["plan_id"] = execution_plan_id_map[agent_output["plan_id"]]
            agent_output["plan_run_id"] = plan_run_id_map[agent_output["plan_run_id"]]
            if not agent_output["task_id"]:
                null_task_id_output_entries.append(agent_output)
            else:
                regular_output_entries.append(agent_output)
        if null_task_id_output_entries:
            to_insert.append(
                InsertToTableArgs(
                    table_name="agent.agent_outputs", rows=null_task_id_output_entries
                )
            )
        if regular_output_entries:
            to_insert.append(
                InsertToTableArgs(table_name="agent.agent_outputs", rows=regular_output_entries)
            )
        await self.pg.insert_atomic(to_insert=to_insert)

    async def insert_agent_qc(self, agent_qc: AgentQC) -> None:
        sql = """
        INSERT INTO agent.agent_qc
          (agent_qc_id, agent_id, plan_id, user_id, query, agent_status, cs_reviewer, eng_reviewer,
           prod_reviewer, follow_up, score_rating, priority, use_case, problem_area,
           cs_failed_reason, cs_attempt_reprompting, cs_expected_output, cs_notes,
           canned_prompt_id, eng_failed_reason, eng_solution, eng_solution_difficulty,
           jira_link, slack_link, fullstory_link, duplicate_agent, created_at, last_updated, query_order, is_spoofed)
        VALUES (%(agent_qc_id)s, %(agent_id)s, %(plan_id)s, %(user_id)s, %(query)s, %(agent_status)s,
                %(cs_reviewer)s, %(eng_reviewer)s, %(prod_reviewer)s, %(follow_up)s,
                %(score_rating)s, %(priority)s, %(use_case)s, %(problem_area)s,
                %(cs_failed_reason)s, %(cs_attempt_reprompting)s, %(cs_expected_output)s, %(cs_notes)s,
                %(canned_prompt_id)s, %(eng_failed_reason)s, %(eng_solution)s, %(eng_solution_difficulty)s,
                %(jira_link)s, %(slack_link)s, %(fullstory_link)s, %(duplicate_agent)s,
                %(created_at)s, %(last_updated)s, %(query_order)s, %(is_spoofed)s)
        """
        await self.pg.generic_write(
            sql,
            {
                "agent_qc_id": agent_qc.agent_qc_id,
                "agent_id": agent_qc.agent_id,
                "plan_id": agent_qc.plan_id,
                "user_id": agent_qc.user_id,
                "query": agent_qc.query,
                "agent_status": agent_qc.agent_status,
                "cs_reviewer": agent_qc.cs_reviewer,
                "eng_reviewer": agent_qc.eng_reviewer,
                "prod_reviewer": agent_qc.prod_reviewer,
                "follow_up": agent_qc.follow_up,
                "score_rating": agent_qc.score_rating,
                "priority": agent_qc.priority,
                "use_case": agent_qc.use_case,
                "problem_area": agent_qc.problem_area,
                "cs_failed_reason": agent_qc.cs_failed_reason,
                "cs_attempt_reprompting": agent_qc.cs_attempt_reprompting,
                "cs_expected_output": agent_qc.cs_expected_output,
                "cs_notes": agent_qc.cs_notes,
                "canned_prompt_id": agent_qc.canned_prompt_id,
                "eng_failed_reason": agent_qc.eng_failed_reason,
                "eng_solution": agent_qc.eng_solution,
                "eng_solution_difficulty": agent_qc.eng_solution_difficulty,
                "jira_link": agent_qc.jira_link,
                "slack_link": agent_qc.slack_link,
                "fullstory_link": agent_qc.fullstory_link,
                "duplicate_agent": agent_qc.duplicate_agent,
                "created_at": agent_qc.created_at,
                "last_updated": agent_qc.last_updated,
                "query_order": agent_qc.query_order,
                "is_spoofed": agent_qc.is_spoofed,
            },
        )

    async def get_agent_qc_id_by_agent_id(
        self, agent_id: str, plan_id: str
    ) -> Tuple[Optional[str], bool]:
        sql = """
        SELECT agent_qc_id::TEXT, is_spoofed
        FROM agent.agent_qc
        WHERE agent_id::TEXT = %(agent_id)s
        AND plan_id::TEXT = %(plan_id)s
        """
        result = await self.pg.generic_read(sql, {"agent_id": agent_id, "plan_id": plan_id})

        # Check if the result is found and return agent_qc_id, else None
        if result:
            return result[0]["agent_qc_id"], result[0]["is_spoofed"]
        return None, False

    async def update_agent_qc(self, agent_qc: AgentQC) -> None:

        # Build a dictionary of only the fields that are not None and not in the immutable fields
        values_to_update = {
            "cs_reviewer": agent_qc.cs_reviewer,
            "eng_reviewer": agent_qc.eng_reviewer,
            "prod_reviewer": agent_qc.prod_reviewer,
            "follow_up": agent_qc.follow_up,
            "score_rating": agent_qc.score_rating,
            "priority": agent_qc.priority,
            "use_case": agent_qc.use_case,
            "problem_area": agent_qc.problem_area,
            "cs_failed_reason": agent_qc.cs_failed_reason,
            "cs_attempt_reprompting": agent_qc.cs_attempt_reprompting,
            "cs_expected_output": agent_qc.cs_expected_output,
            "cs_notes": agent_qc.cs_notes,
            "canned_prompt_id": agent_qc.canned_prompt_id,
            "eng_failed_reason": agent_qc.eng_failed_reason,
            "eng_solution": agent_qc.eng_solution,
            "eng_solution_difficulty": agent_qc.eng_solution_difficulty,
            "jira_link": agent_qc.jira_link,
            "slack_link": agent_qc.slack_link,
            "fullstory_link": agent_qc.fullstory_link,
            "duplicate_agent": agent_qc.duplicate_agent,
            "last_updated": agent_qc.last_updated,
            "cs_reviewed": agent_qc.cs_reviewed,
            "eng_reviewed": agent_qc.eng_reviewed,
            "prod_reviewed": agent_qc.prod_reviewed,
            "prod_priority": agent_qc.prod_priority,
            "prod_notes": agent_qc.prod_notes,
        }

        # Remove fields that are None or immutable
        values_to_update = {
            key: value for key, value in values_to_update.items() if value is not None
        }

        # Perform the update only with mutable fields
        await self.pg.generic_update(
            table_name="agent.agent_qc",
            where={"agent_qc_id": agent_qc.agent_qc_id},
            values_to_update=values_to_update,
        )

    async def boosted_internal_users(self) -> List[str]:
        """
        Returns the list of Internal Boosted Users
        """
        BOOSTED_ORG_ID_DEV = "40bb9e63-5719-443c-aad1-3270c0d237e5"
        BOOSTED_ORG_ID_PROD = "101f7a87-9138-4458-8e0d-618a8e11826d"

        sql = """
            SELECT u.id FROM user_service.users u
            LEFT JOIN user_service.company_membership cm ON u.id::TEXT = cm.user_id::TEXT
            WHERE cm.company_id = %(boosted_org_id)s
        """

        env = get_environment_tag()
        is_dev = env in (DEV_TAG, LOCAL_TAG)

        users = await self.pg.generic_read(
            sql, params={"boosted_org_id": BOOSTED_ORG_ID_DEV if is_dev else BOOSTED_ORG_ID_PROD}
        )

        return [str(user["id"]) for user in users]

    async def search_agent_qc(
        self,
        filter_criteria: List[HorizonCriteria],
        search_criteria: List[HorizonCriteria],
        pagination: Optional[Pagination] = None,
    ) -> Tuple[List[AgentQC], int]:
        # Need to do complicated users join since some users belong to more than one company in
        #   user_service.company_membership, resulting in duplicate rows, so this will prevent
        #   erroneous join with company_membership for internal users
        sql = """
        SELECT
            COUNT(*) over () as total_agent_qcs,
            aqc.agent_qc_id::TEXT, aqc.agent_id::TEXT, ag.agent_name, aqc.plan_id::TEXT, aqc.user_id::TEXT,
            aqc.query, aqc.agent_status,
            aqc.cs_reviewer::TEXT, aqc.eng_reviewer::TEXT, aqc.prod_reviewer::TEXT, aqc.follow_up, aqc.score_rating,
            aqc.priority, aqc.use_case, aqc.problem_area, aqc.cs_failed_reason, aqc.cs_attempt_reprompting,
            aqc.cs_expected_output, aqc.cs_notes, aqc.canned_prompt_id, aqc.eng_failed_reason, aqc.eng_solution,
            aqc.eng_solution_difficulty, aqc.jira_link, aqc.slack_link, aqc.fullstory_link, aqc.duplicate_agent::TEXT,
            aqc.created_at, aqc.last_updated, aqc.cs_reviewed, aqc.eng_reviewed, aqc.prod_reviewed,
            aqc.prod_priority, aqc.prod_notes, aqc.is_spoofed,
            users.cognito_username, users.owner_name AS owner_name,
            users.owner_organization_name AS owner_organization_name,
            json_agg(af.*) AS agent_feedbacks
        FROM agent.agent_qc aqc
        LEFT JOIN agent.agents ag ON aqc.agent_id = ag.agent_id
        LEFT JOIN (
            SELECT u.id, u.cognito_username, u.name AS owner_name, org.name AS owner_organization_name
            FROM user_service.users u
                LEFT JOIN user_service.company_membership cm ON u.id::TEXT = cm.user_id::TEXT
                LEFT JOIN user_service.organizations org ON cm.company_id = org.id::TEXT
                WHERE NOT u.id::TEXT = ANY(%(boosted_user_ids)s)
            UNION
            SELECT u.id, u.cognito_username, u.name as owner_name, 'Boosted.ai' as owner_organization_name
            FROM user_service.users u
                WHERE u.id::TEXT = ANY(%(boosted_user_ids)s)
        ) AS users ON users.id::TEXT = aqc.user_id::TEXT
        LEFT JOIN agent.feedback af ON aqc.agent_id = af.agent_id AND aqc.plan_id = af.plan_id
        """

        def criteria_to_sql_clause_helper(
            criterion: HorizonCriteria, params: Dict[str, Any], param1_name: str, param2_name: str
        ) -> str:
            if criterion.operator == "BETWEEN":
                params[param1_name] = criterion.arg1
                params[param2_name] = criterion.arg2
                return f" {criterion.column} BETWEEN %({param1_name})s AND %({param2_name})s"
            elif criterion.operator == "ILIKE":
                params[param1_name] = f"%{criterion.arg1}%"  # Add wildcards for partial matches
                return f" {criterion.column} ILIKE %({param1_name})s"
            elif criterion.operator == "=":
                params[param1_name] = criterion.arg1
                return f" {criterion.column} = %({param1_name})s"
            elif criterion.operator == "IN" and isinstance(criterion.arg1, list):
                params[param1_name] = criterion.arg1  # Ensure arg1 is a list for IN queries
                return f" {criterion.column} = ANY(%({param1_name})s)"
            elif criterion.operator == "IN":  # Fallback to EQUAL behavior if arg1 is not a list
                params[param1_name] = criterion.arg1
                return f" {criterion.column} = %({param1_name})s"
            elif criterion.operator == ">":
                params[param1_name] = criterion.arg1
                return f" {criterion.column} > %({param1_name})s"
            elif criterion.operator == "<":
                params[param1_name] = criterion.arg1
                return f" {criterion.column} < %({param1_name})s"
            elif criterion.operator == "!=":
                params[param1_name] = criterion.arg1
                return f" {criterion.column} != %({param1_name})s"
            elif criterion.operator == "=ANY":
                params[param1_name] = criterion.arg1
                return f" {criterion.column} = ANY(%({param1_name})s)"

            logger.warning(f"Unknown operator {criterion.operator=}")
            return ""

        boosted_user_ids = await self.boosted_internal_users()

        sql_params: Dict[str, Any] = {"boosted_user_ids": boosted_user_ids}
        if len(filter_criteria) > 0 or len(search_criteria) > 0:
            sql += " WHERE "

        # Dynamically add conditions based on HorizonCriteria, use AND between conditions
        for idx, criteria in enumerate(filter_criteria):
            if idx != 0:
                sql += " AND"
            param1 = f"filter_arg1_{idx}"
            param2 = f"filter_arg2_{idx}"
            sql += criteria_to_sql_clause_helper(criteria, sql_params, param1, param2)

        # Use OR between conditions
        for idx, criteria in enumerate(search_criteria):
            if idx == 0 and len(filter_criteria) > 0:
                sql += " AND ("
            elif idx != 0:
                sql += " OR"
            param1 = f"search_arg1_{idx}"
            param2 = f"search_arg2_{idx}"
            sql += criteria_to_sql_clause_helper(criteria, sql_params, param1, param2)

        if len(filter_criteria) and len(search_criteria) > 0:
            sql += ")"

        # Always sort by latest agent ran
        sql += """
            GROUP BY aqc.agent_qc_id, users.cognito_username, users.owner_name,
                users.owner_organization_name, ag.agent_name
            ORDER BY aqc.created_at desc
        """

        if pagination:
            sql += " LIMIT %(page_size)s OFFSET %(page_index)s"
            sql_params["page_size"] = pagination.page_size
            sql_params["page_index"] = pagination.page_size * pagination.page_index

        records = await self.pg.generic_read(sql, sql_params)
        agent_qcs = []
        total_agent_qcs = 0
        for record in records:
            total_agent_qcs = record.pop("total_agent_qcs", 0)
            agent_feedbacks = [
                AgentFeedback(**agent_feedback)
                for agent_feedback in record.pop("agent_feedbacks", [])
                if agent_feedback is not None
            ]
            agent_qcs.append(AgentQC(agent_feedbacks=agent_feedbacks, **record))
        return agent_qcs, total_agent_qcs

    async def get_agent_qc_by_ids(self, agent_qc_ids: List[str]) -> list[AgentQC]:
        ids_filter = [
            HorizonCriteria(
                column="aqc.agent_qc_id",
                operator=HorizonCriteriaOperator.equal_any,
                arg1=agent_qc_ids,
                arg2=None,
            )
        ]

        # Call the search DB function to retrieve agent QC by IDs
        agent_qcs, _ = await self.search_agent_qc(ids_filter, [], None)

        # Return the list of AgentQC objects
        return agent_qcs

    async def get_agent_qc_by_user_ids(self, user_ids: List[str]) -> List[AgentQC]:
        user_ids_filter = [
            HorizonCriteria(
                column="aqc.user_id",
                operator=HorizonCriteriaOperator.equal_any,
                arg1=user_ids,
                arg2=None,
            )
        ]

        # Call the search DB function to retrieve agent QCs by user_id
        agent_qcs, _ = await self.search_agent_qc(user_ids_filter, [], None)

        # Return the list of AgentQC objects
        return agent_qcs

    @async_perf_logger
    async def get_agent_metadata_for_qc(
        self,
        live_only: bool = True,
        start_dt: Optional[datetime.datetime] = None,
        end_dt: Optional[datetime.datetime] = None,
        filter_deleted: bool = True,
    ) -> List[AgentQCInfo]:
        where_parts = []
        where_clause = ""
        params: Dict[str, Any] = {}
        if filter_deleted:
            where_parts.append("NOT a.deleted")
        if live_only:
            where_parts.append("a.automation_enabled")
        if where_parts:
            parts_str = " AND ".join(where_parts)
            where_clause = f"WHERE {parts_str}"

        sql = f"""
        SELECT DISTINCT ON (a.agent_id)
          a.agent_id::TEXT, a.agent_name, a.user_id::TEXT, u.name AS user_name, a.help_requested,
          o.id::TEXT AS user_org_id, o.name AS user_org_name, (NOT is_client) AS user_is_internal,
          JSON_AGG(
            JSON_BUILD_OBJECT(
              'plan_run_id', pr.plan_run_id,
              'status', pr.status,
              'created_at', pr.created_at
            )
          ) AS plan_run_info
        FROM agent.agents a
        JOIN agent.plan_runs pr ON pr.agent_id = a.agent_id
        JOIN user_service.users u ON a.user_id = u.id
        JOIN user_service.company_membership cm ON cm.user_id = u.id::TEXT
        JOIN user_service.organizations o ON o.id::TEXT = cm.company_id
        {where_clause}
        GROUP BY a.agent_id, a.agent_name, a.user_id, user_name, user_org_id, user_org_name,
                 user_is_internal
        """

        qc_infos = []
        start = time.time()
        results = await self.pg.generic_read(sql, params)
        logger.info(f"Fetch agent data for QC in {time.time() - start}")
        for row in results:
            if not row["plan_run_info"]:
                continue
            row["plan_run_info"] = sorted(
                row["plan_run_info"], reverse=True, key=lambda run: run["created_at"]
            )
            most_recent_run = row["plan_run_info"][0]
            info = AgentQCInfo(
                agent_id=row["agent_id"],
                agent_name=row["agent_name"],
                help_requested=row["help_requested"],
                user_id=row["user_id"],
                user_name=row["user_name"],
                user_org_id=row["user_org_id"],
                user_org_name=row["user_org_name"],
                user_is_internal=row["user_is_internal"],
                most_recent_plan_run_id=most_recent_run["plan_run_id"],
                most_recent_plan_run_status=Status(
                    most_recent_run["status"] or Status.COMPLETE.value
                ),
                last_run_start=parse(most_recent_run["created_at"]),
            )
            if start_dt and info.last_run_start < start_dt:
                continue
            if end_dt and info.last_run_start > end_dt:
                continue
            last_successful_run = None
            run_count_by_status: Dict[Status, int] = defaultdict(int)
            for run_info in row["plan_run_info"]:
                if not last_successful_run and run_info["status"] == Status.COMPLETE:
                    last_successful_run = parse(run_info["created_at"])
                run_count_by_status[Status(run_info["status"] or Status.COMPLETE.value)] += 1
            info.run_count_by_status = run_count_by_status
            info.last_successful_run = last_successful_run
            qc_infos.append(info)
        return qc_infos

    async def get_company_descriptions(self, gbi_ids: List[int]) -> Dict[int, str]:
        sql = """
        SELECT DISTINCT ON (ssm.gbi_id) ssm.gbi_id, cds.company_description_short, cds.last_updated
        FROM spiq_security_mapping ssm
        JOIN nlp_service.company_descriptions_short cds
        ON cds.spiq_company_id = ssm.spiq_company_id
        WHERE ssm.gbi_id = ANY(%(gbi_ids)s)
        ORDER BY ssm.gbi_id, cds.last_updated DESC NULLS LAST;
        """

        long_sql = sql.replace("_short", "")

        rows, rows_long = await asyncio.gather(
            self.pg.generic_read(sql, {"gbi_ids": gbi_ids}),
            self.pg.generic_read(long_sql, {"gbi_ids": gbi_ids}),
        )

        descriptions_rows = {row["gbi_id"]: row["company_description_short"] for row in rows}
        descriptions = {gbi: descriptions_rows.get(gbi, "No description found") for gbi in gbi_ids}

        # replace with long if it exists
        for row in rows_long:
            descriptions[row["gbi_id"]] = row["company_description"]
        return descriptions

    async def get_short_company_descriptions_for_gbi_ids(
        self, gbi_ids: List[int]
    ) -> Dict[int, Tuple[Optional[str], Optional[datetime.datetime]]]:
        """
        Given a list of gbi ids, return a mapping from gbi id to its company
        description and last updated time. If the long description isn't
        present, fall back to the short description.
        """
        sql = """
        SELECT
          ssm.gbi_id,
          cds.company_description_short AS company_description,
          cds.last_updated AT TIME ZONE 'UTC' AS last_updated
        FROM spiq_security_mapping ssm
        LEFT JOIN nlp_service.company_descriptions_short cds
          ON ssm.spiq_company_id = cds.spiq_company_id
        WHERE ssm.gbi_id = ANY(%s);
        """
        records = await self.generic_read(sql, params=[gbi_ids])
        return {r["gbi_id"]: (r["company_description"], r["last_updated"]) for r in records}

    @async_perf_logger
    async def is_user_internal(self, user_id: str) -> bool:
        sql = """
        SELECT
            (NOT o.is_client) AS user_is_internal
        FROM user_service.users u
        JOIN user_service.company_membership cm ON cm.user_id = u.id::TEXT
        JOIN user_service.organizations o ON o.id::TEXT = cm.company_id
        WHERE u.id = %(user_id)s
        """

        params = {"user_id": user_id}

        start = time.time()
        result = await self.pg.generic_read(sql, params)
        logger.info(f"Fetch internal user status for user_id {user_id} in {time.time() - start}")

        # Some users belong to more than one company in the table, need to loop to determine if any are internal
        for res in result:
            if res["user_is_internal"]:
                return True

        if not result:
            logger.warning(f"is_user_internal : Failed to find row for: {user_id} , assuming False")

        return False

    async def get_plan_ids(self, plan_run_ids: List[str]) -> Dict[str, str]:
        sql = """
        SELECT plan_run_id::TEXT, plan_id::TEXT from agent.plan_runs where plan_run_id = ANY(%(plan_run_ids)s)
        """
        rows = await self.pg.generic_read(sql, {"plan_run_ids": plan_run_ids})

        return {row["plan_run_id"]: row["plan_id"] for row in rows}


async def get_chat_history_from_db(agent_id: str, db: Union[AsyncDB, Postgres]) -> ChatContext:
    if isinstance(db, Postgres):
        return db.get_chats_history_for_agent(agent_id)
    elif isinstance(db, AsyncDB):
        return await db.get_chats_history_for_agent(agent_id)


async def get_latest_execution_plan_from_db(agent_id: str, db: Union[AsyncDB, Postgres]) -> Tuple[
    Optional[str],
    Optional[ExecutionPlan],
    Optional[datetime.datetime],
    Optional[str],
    Optional[str],
]:
    if isinstance(db, Postgres):
        return db.get_latest_execution_plan(agent_id)
    elif isinstance(db, AsyncDB):
        return await db.get_latest_execution_plan(agent_id)
    else:
        return None, None, None, None, None


ASYNC_DB: Optional[AsyncDB] = None
ASYNC_RO_DB: Optional[AsyncDB] = None


def get_async_db(
    sync_db: bool = False,
    skip_commit: bool = False,
    read_only: bool = False,
    min_pool_size: int = 1,
    max_pool_size: int = 8,
) -> AsyncDB:
    """Get the single AsyncDB instance based on the parameters.

    Args:
        sync_db (bool, optional): Whether to use the SyncBoostedPG or AsyncPostgresBase under the hood.
        skip_commit (bool, optional): Whether to skip the commit after each write operation. Note
    that if it's True, it will use the SyncBoostedPG under the hood (because AsyncPostgresBase does
    not support skipping commit).
    """
    global ASYNC_DB, ASYNC_RO_DB
    if read_only:
        if ASYNC_RO_DB is None:
            if sync_db:
                ASYNC_RO_DB = AsyncDB(pg=SyncBoostedPG(read_only=read_only))
            else:
                ASYNC_RO_DB = AsyncDB(
                    pg=AsyncPostgresBase(
                        read_only=read_only,
                        min_pool_size=min_pool_size,
                        max_pool_size=max_pool_size,
                    )
                )
        return ASYNC_RO_DB

    if ASYNC_DB is None:
        if sync_db or skip_commit:
            ASYNC_DB = AsyncDB(pg=SyncBoostedPG(skip_commit=skip_commit))
        else:
            ASYNC_DB = AsyncDB(
                pg=AsyncPostgresBase(
                    min_pool_size=min_pool_size,
                    max_pool_size=max_pool_size,
                )
            )
    else:
        if skip_commit:
            if (
                not isinstance(ASYNC_DB.pg, SyncBoostedPG)
                or ASYNC_DB.pg.db.skip_commit != skip_commit
            ):
                ASYNC_DB = AsyncDB(pg=SyncBoostedPG(skip_commit=skip_commit))
        else:
            if sync_db and not isinstance(ASYNC_DB.pg, SyncBoostedPG):
                ASYNC_DB = AsyncDB(pg=SyncBoostedPG(skip_commit=skip_commit))
            elif not sync_db and not isinstance(ASYNC_DB.pg, AsyncPostgresBase):
                ASYNC_DB = AsyncDB(
                    pg=AsyncPostgresBase(
                        min_pool_size=min_pool_size,
                        max_pool_size=max_pool_size,
                    )
                )

    return ASYNC_DB
