import asyncio
import datetime
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from agent_service.endpoints.models import (
    Account,
    AgentFeedback,
    AgentMetadata,
    AgentNotificationEmail,
    AgentOutput,
    AgentQC,
    AgentSchedule,
    CustomNotification,
    HorizonCriteria,
    PlanRunStatusInfo,
    SetAgentFeedBackRequest,
    Status,
    TaskRunStatusInfo,
    TaskStatus,
)
from agent_service.io_type_utils import IOType, dump_io_type, load_io_type

# Make sure all io_types are registered
from agent_service.io_types import *  # noqa
from agent_service.io_types.graph import GraphOutput
from agent_service.io_types.table import TableOutput
from agent_service.io_types.text import TextOutput
from agent_service.planner.planner_types import ExecutionPlan, PlanStatus, RunMetadata
from agent_service.types import ChatContext, Message, Notification, PlanRunContext
from agent_service.utils.async_postgres_base import AsyncPostgresBase
from agent_service.utils.boosted_pg import BoostedPG, InsertToTableArgs
from agent_service.utils.cache_utils import CacheBackend
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import Postgres, SyncBoostedPG
from agent_service.utils.prompt_template import PromptTemplate
from agent_service.utils.sidebar_sections import SidebarSection


class AsyncDB:
    def __init__(self, pg: BoostedPG):
        self.pg = pg

    async def get_prev_outputs_for_agent_plan(
        self, agent_id: str, plan_id: str, latest_plan_run_id: str
    ) -> Optional[Tuple[List[IOType], datetime.datetime]]:
        """
        Returns the prior list of outputs for a plan, as well as the date the outputs were created.
        """
        sql = """
        SELECT plan_run_id,
          MAX(ao.created_at) AS prev_date,
          ARRAY_AGG(ao.output ORDER BY ao.created_at) AS outputs
        FROM agent.agent_outputs ao
        WHERE agent_id = %(agent_id)s AND "output" NOTNULL AND is_intermediate = FALSE
                AND NOT ao.deleted
                AND plan_id = %(plan_id)s
                AND plan_run_id IN
                  (
                    SELECT plan_run_id FROM agent.plan_runs
                    WHERE plan_run_id != %(latest_plan_run_id)s
                      AND agent_id = %(agent_id)s
                      AND plan_id = %(plan_id)s
                    ORDER BY created_at DESC
                    LIMIT 1
                  )
        GROUP BY plan_run_id
        LIMIT 1
        """
        rows = await self.pg.generic_read(
            sql,
            {"agent_id": agent_id, "plan_id": plan_id, "latest_plan_run_id": latest_plan_run_id},
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
    ) -> Optional[str]:
        """
        Returns the last plan run for the agent before the latest_run
        """
        date_filter = ""
        if cutoff_dt:
            date_filter = " AND created_at < %(cutoff_dt)s"

        sql = f"""
        SELECT plan_run_id FROM agent.plan_runs
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
            return rows[0]["plan_run_id"]
        else:
            return None

    async def get_agent_outputs(
        self, agent_id: str, plan_run_id: Optional[str] = None, cache: Optional[CacheBackend] = None
    ) -> List[AgentOutput]:
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

        sql = f"""
                SELECT ao.plan_id::VARCHAR, ao.output_id::VARCHAR, ao.plan_run_id::VARCHAR,
                    ao.task_id::VARCHAR,
                    ao.is_intermediate, ao.live_plan_output,
                    ao.output, ao.created_at, pr.shared, pr.run_metadata,
                    ao.plan_id::TEXT, ep.plan, ep.locked_tasks
                FROM agent.agent_outputs ao
                LEFT JOIN agent.plan_runs pr
                  ON ao.plan_run_id = pr.plan_run_id
                LEFT JOIN agent.execution_plans ep
                  ON ao.plan_id = ep.plan_id
                WHERE {where_clause}
                ORDER BY created_at ASC;
                """
        rows = await self.pg.generic_read(sql, params)
        if not rows:
            return []

        async def get_output_id_from_cache(
            output_id: str,
        ) -> Tuple[str, Optional[Union[TextOutput, GraphOutput, TableOutput]]]:
            cached_output: Optional[Union[TextOutput, GraphOutput, TableOutput]] = None
            if cache:
                cached_output = await cache.get(output_id)  # type: ignore
            return output_id, cached_output

        async def get_output_values() -> List[Union[TextOutput, GraphOutput, TableOutput]]:
            cached_output_ids = set()
            to_return = []
            if cache:
                output_get_tasks = [get_output_id_from_cache(row["output_id"]) for row in rows]
                outputs_with_ids = await asyncio.gather(*output_get_tasks)
                for cached_output_id, cached_output in outputs_with_ids:
                    if cached_output:
                        cached_output_ids.add(cached_output_id)
                        to_return.append(cached_output)

            non_cached_output_tasks = []
            for row in rows:

                async def get_non_cached(output_id: str, a_io_output: IOType) -> Any:
                    res = await get_output_from_io_type(a_io_output, pg=self.pg)
                    if cache:
                        await cache.set(key=output_id, val=res)
                    return res

                if row["output_id"] not in cached_output_ids:
                    io_output = load_io_type(row["output"]) if row["output"] else row["output"]
                    non_cached_output_tasks.append(get_non_cached(row["output_id"], io_output))
            if non_cached_output_tasks:
                results = await asyncio.gather(*non_cached_output_tasks)
                to_return.extend(results)
            return to_return

        output_values = await get_output_values()

        outputs = []
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
                plan = ExecutionPlan.model_validate(row["plan"])
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

        await self.pg.multi_row_insert(table_name="agent.cancelled_ids", rows=cancelled_ids)

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
        self, agent_id: str, task_ids: List[str], plan_id: str
    ) -> Dict[str, str]:
        """
        Returns a dict from task_id to log_id.
        """
        sql = """
        SELECT DISTINCT ON (task_id) log_id::TEXT, task_id::TEXT
        FROM agent.work_logs
        WHERE agent_id = %(agent_id)s AND plan_id = %(plan_id)s
              AND task_id = ANY(%(task_ids)s)
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
        plan = ExecutionPlan.model_validate(rows[0]["plan"])
        plan.locked_task_ids = list(rows[0]["locked_tasks"])
        return (
            rows[0]["plan_id"],
            plan,
            rows[0]["created_at"],
            rows[0]["status"],
            rows[0]["upcoming_plan_run_id"],
        )

    async def get_execution_plan_for_run(self, plan_run_id: str) -> ExecutionPlan:
        sql = """
            SELECT plan
            FROM agent.execution_plans ep
            JOIN agent.plan_runs pr ON ep.plan_id = pr.plan_id
            WHERE pr.plan_run_id = %(plan_run_id)s
        """
        rows = await self.pg.generic_read(sql, params={"plan_run_id": plan_run_id})
        return ExecutionPlan.model_validate(rows[0]["plan"])

    async def get_agent_owner(self, agent_id: str) -> Optional[str]:
        """
        This function retrieves the owner of an agent, mainly used in authorization.
        Caches the result for 128 calls since the owner cannot change.

        Args:
            agent_id: The agent id to retrieve the owner for.

        Returns: The user id of the agent owner.
        """
        sql = """
            SELECT user_id::VARCHAR
            FROM agent.agents
            WHERE agent_id = %(agent_id)s;
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
            plan = ExecutionPlan.model_validate(row["plan"])
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
              COALESCE(nf.unread, FALSE) as unread
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

    async def create_agent(self, agent_metadata: AgentMetadata) -> None:
        await self.pg.multi_row_insert(
            table_name="agent.agents", rows=[agent_metadata.to_agent_row()]
        )

    async def update_agent_draft_status(self, agent_id: str, is_draft: bool) -> None:
        sql = """
        UPDATE agent.agents SET is_draft = %(is_draft)s WHERE agent_id = %(agent_id)s
        """
        await self.pg.generic_write(sql, params={"agent_id": agent_id, "is_draft": is_draft})

    async def update_execution_plan_status(
        self, plan_id: str, agent_id: str, status: PlanStatus = PlanStatus.READY
    ) -> None:
        sql = """
        INSERT INTO agent.execution_plans (plan_id, agent_id, plan, created_at, last_updated, status)
        VALUES (
          %(plan_id)s, %(agent_id)s, %(empty_plan)s, %(created_at)s, %(last_updated)s, %(status)s
        )
        ON CONFLICT (plan_id) DO UPDATE SET
          last_updated = NOW(),
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
            },
        )

    async def write_execution_plan(
        self,
        plan_id: str,
        agent_id: str,
        plan: ExecutionPlan,
        status: PlanStatus = PlanStatus.READY,
    ) -> None:
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
          last_updated = NOW(),
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
            },
        )

    async def update_plan_run(
        self, agent_id: str, plan_id: str, plan_run_id: str, status: Status = Status.NOT_STARTED
    ) -> None:
        sql = """
        INSERT INTO agent.plan_runs (agent_id, plan_id, plan_run_id, status)
        VALUES (%(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(status)s)
        ON CONFLICT (plan_run_id) DO UPDATE SET
          agent_id = EXCLUDED.agent_id,
          plan_id = EXCLUDED.plan_id,
          status = EXCLUDED.status,
          last_updated = NOW()
        """
        await self.pg.generic_write(
            sql,
            params={
                "agent_id": agent_id,
                "plan_id": plan_id,
                "plan_run_id": plan_run_id,
                "status": status.value,
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
        debug_info: Optional[str] = None,
    ) -> None:
        sql = """
        INSERT INTO agent.task_run_info (task_id, agent_id, plan_run_id,
          tool_name, task_args, debug_info, output)
        VALUES (%(task_id)s, %(agent_id)s, %(plan_run_id)s,
          %(tool_name)s, %(task_args)s, %(debug_info)s, %(output)s)
        """
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
            },
        )

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
        sql = """
        INSERT INTO agent.task_runs AS tr (task_id, agent_id, plan_run_id, status)
        VALUES (%(task_id)s, %(agent_id)s, %(plan_run_id)s, %(status)s)
        ON CONFLICT (plan_run_id, task_id) DO UPDATE SET
          agent_id = EXCLUDED.agent_id,
          status = EXCLUDED.status,
          last_updated = NOW() WHERE tr.status != EXCLUDED.status
        """
        params = [
            {
                "agent_id": agent_id,
                "task_id": task.task_id,
                "plan_run_id": plan_run_id,
                "status": task.status.value,
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

    @async_perf_logger
    async def get_user_all_agents(
        self, user_id: Optional[str] = None, agent_ids: Optional[List[str]] = None
    ) -> List[AgentMetadata]:
        """
        This function retrieves all agents for a given user, optionally filtered
        by a list of agent ids.

        Args:
            user_id: The user id to retrieve agents for.
            agent_ids: The list of agent id's to filter by

        Returns: A list of all agents for the user, optionally filtered.
        """
        agent_where_clauses = ["not deleted"]
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
        WITH a_id AS
          (
            SELECT a.agent_id, a.user_id, agent_name, a.created_at,
              a.last_updated, a.automation_enabled, a.schedule, a.section_id, a.deleted, a.is_draft
             FROM agent.agents a
             {agent_where_clause}
          ),
          lr AS
          (
            SELECT DISTINCT ON (pr.agent_id) pr.agent_id, pr.created_at, pr.run_metadata
            FROM agent.plan_runs pr
            ORDER BY pr.agent_id, pr.created_at DESC
          ),
          msg AS
          (
            SELECT DISTINCT ON (m.agent_id) m.agent_id, m.message
            FROM agent.chat_messages m
            ORDER BY m.agent_id, m.message_time DESC
          ),
          nu AS
          (
            SELECT n.agent_id, COUNT(*) AS num_unread
            FROM agent.notifications n
            WHERE n.unread
                GROUP BY n.agent_id
          ),
          lo AS
          (
            SELECT DISTINCT ON (ao.agent_id) ao.agent_id, ao.created_at
            FROM agent.agent_outputs ao
            WHERE NOT ao.deleted
            ORDER BY ao.agent_id, ao.created_at DESC
          )
          SELECT a_id.agent_id::VARCHAR, a_id.user_id::VARCHAR, a_id.agent_name, a_id.created_at,
            a_id.last_updated, a_id.automation_enabled, a_id.section_id::VARCHAR, lr.created_at AS last_run,
            msg.message AS latest_agent_message, nu.num_unread AS unread_notification_count,
            a_id.schedule, lr.run_metadata, a_id.deleted, a_id.is_draft, lo.created_at AS output_last_updated
          FROM a_id
          LEFT JOIN lr ON lr.agent_id = a_id.agent_id
          LEFT JOIN msg ON msg.agent_id = a_id.agent_id
          LEFT JOIN nu ON nu.agent_id = a_id.agent_id
          LEFT JOIN lo ON lo.agent_id = a_id.agent_id
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
                AgentMetadata(
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
        self, agent_id: str, emails_to_user: Dict[str, Account], delete_previous_emails: bool = True
    ) -> None:
        records_to_upload = []
        already_seen = set()
        # remove all the subs before adding new ones to avoid
        # duplicates, when calling from enable_agent_automation
        # need to make sure we do not remove all the emails
        if delete_previous_emails:
            await self.delete_all_email_subscriptions_for_agent(agent_id)
        for email, user in emails_to_user.items():
            if email not in already_seen:
                already_seen.add(email)
                row = {"agent_id": agent_id, "email": email, "user_id": user.user_id}
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

    async def set_prompt_template_visibility(self, template_id: str, is_visible: bool) -> None:
        sql = """
        UPDATE agent.prompt_templates
        SET is_visible = %(is_visible)s
        WHERE template_id = %(template_id)s
        """
        await self.pg.generic_write(
            sql,
            params={"template_id": template_id, "is_visible": is_visible},
        )

    async def update_prompt_template(self, prompt_template: PromptTemplate) -> None:
        sql = """
        UPDATE agent.prompt_templates
        SET name = %(name)s, description = %(description)s, prompt = %(prompt)s,
            category = %(category)s, plan = %(plan)s, is_visible = %(is_visible)s, created_at = %(created_at)s
        WHERE template_id = %(template_id)s
        """
        await self.pg.generic_write(
            sql,
            {
                "template_id": prompt_template.template_id,
                "name": prompt_template.name,
                "description": prompt_template.description,
                "prompt": prompt_template.prompt,
                "category": prompt_template.category,
                "plan": prompt_template.plan.model_dump_json(),
                "is_visible": prompt_template.is_visible,
                "created_at": prompt_template.created_at,
            },
        )

    async def insert_prompt_template(self, prompt_template: PromptTemplate) -> None:
        sql = """
        INSERT INTO agent.prompt_templates
          (template_id, name, description, prompt, category, created_at, plan, is_visible)
        VALUES (%(template_id)s, %(name)s, %(description)s, %(prompt)s,
                %(category)s, %(created_at)s, %(plan)s, %(is_visible)s)
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
                "is_visible": prompt_template.is_visible,
            },
        )

    async def get_prompt_templates(self) -> List[PromptTemplate]:
        sql = """
        SELECT template_id::TEXT, name, description, prompt, category, created_at, plan, is_visible
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
                    is_visible=row["is_visible"],
                    plan=ExecutionPlan.model_validate(row["plan"]),
                )
            )

        return res

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
           jira_link, slack_link, fullstory_link, duplicate_agent, created_at, last_updated, query_order)
        VALUES (%(agent_qc_id)s, %(agent_id)s, %(plan_id)s, %(user_id)s, %(query)s, %(agent_status)s,
                %(cs_reviewer)s, %(eng_reviewer)s, %(prod_reviewer)s, %(follow_up)s,
                %(score_rating)s, %(priority)s, %(use_case)s, %(problem_area)s,
                %(cs_failed_reason)s, %(cs_attempt_reprompting)s, %(cs_expected_output)s, %(cs_notes)s,
                %(canned_prompt_id)s, %(eng_failed_reason)s, %(eng_solution)s, %(eng_solution_difficulty)s,
                %(jira_link)s, %(slack_link)s, %(fullstory_link)s, %(duplicate_agent)s,
                %(created_at)s, %(last_updated)s, %(query_order)s )
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
            },
        )

    async def get_agent_qc_id_by_agent_id(self, agent_id: str, plan_id: str) -> Optional[str]:
        sql = """
        SELECT agent_qc_id::TEXT
        FROM agent.agent_qc
        WHERE agent_id::TEXT = %(agent_id)s
        AND plan_id::TEXT = %(plan_id)s
        """
        result = await self.pg.generic_read(sql, {"agent_id": agent_id, "plan_id": plan_id})

        # Check if the result is found and return agent_qc_id, else None
        if result:
            return result[0]["agent_qc_id"]
        return None

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

    async def get_agent_qc_by_id(self, ids: list[str]) -> list[AgentQC]:
        sql = """
        SELECT aq.agent_qc_id::TEXT, aq.agent_id::TEXT, ag.agent_name, aq.plan_id::TEXT, aq.user_id::TEXT, aq.query,
               aq.agent_status, aq.cs_reviewer::TEXT, aq.eng_reviewer::TEXT, aq.prod_reviewer::TEXT,
               aq.follow_up, aq.score_rating, aq.priority, aq.use_case,
               aq.problem_area, aq.cs_failed_reason, aq.cs_attempt_reprompting, aq.cs_expected_output, aq.cs_notes,
               aq.canned_prompt_id, aq.eng_failed_reason, aq.eng_solution, aq.eng_solution_difficulty,
               aq.jira_link, aq.slack_link, aq.fullstory_link, aq.duplicate_agent::TEXT, aq.created_at, aq.last_updated,
               us.cognito_username,
               ARRAY_AGG(af.*) AS agent_feedbacks,
               (SELECT tl.cloudwatch_url
                FROM boosted_dag.task_log tl
                WHERE tl.agent_id = aq.agent_id::TEXT
                ORDER BY tl.finished_at DESC
                LIMIT 1) AS cloudwatch_url
        FROM agent.agent_qc aq
        LEFT JOIN user_service.users us ON aq.user_id = us.id
        LEFT JOIN agent.feedback af ON aq.agent_id = af.agent_id AND aq.plan_id = af.plan_id
        LEFT JOIN agent.agents ag ON ag.agent_id = aq.agent_id
        WHERE aq.agent_qc_id = ANY(%(ids)s)
        GROUP BY aq.agent_qc_id, us.cognito_username, ag.agent_name
        """
        records = await self.pg.generic_read(sql, {"ids": ids})
        return [AgentQC(**record) for record in records]

    async def get_agent_qc_by_user(self, user_id: str) -> list[AgentQC]:
        sql = """
        SELECT
            aq.agent_qc_id::TEXT, aq.agent_id::TEXT, ag.agent_name, aq.plan_id::TEXT, aq.user_id::TEXT, aq.query,
           aq.agent_status, aq.cs_reviewer::TEXT, aq.eng_reviewer::TEXT, aq.prod_reviewer::TEXT,
           aq.follow_up, aq.score_rating, aq.priority, aq.use_case,
           aq.problem_area, aq.cs_failed_reason, aq.cs_attempt_reprompting, aq.cs_expected_output, aq.cs_notes,
           aq.canned_prompt_id, aq.eng_failed_reason, aq.eng_solution, aq.eng_solution_difficulty,
           aq.jira_link, aq.slack_link, aq.fullstory_link, aq.duplicate_agent::TEXT, aq.created_at, aq.last_updated,
           us.cognito_username,
            array_agg(af.*) AS agent_feedbacks,
            (SELECT tl.cloudwatch_url
             FROM boosted_dag.task_log tl
             WHERE tl.agent_id = aq.agent_id::TEXT
             ORDER BY tl.finished_at DESC
             LIMIT 1) AS cloudwatch_url
        FROM agent.agent_qc aq
        LEFT JOIN user_service.users us ON aq.user_id = us.id
        LEFT JOIN agent.feedback af ON aq.agent_id = af.agent_id AND aq.plan_id = af.plan_id
        LEFT JOIN agent.agents ag ON aq.agent_id = ag.agent_id
        WHERE aq.user_id = %(user_id)s
        GROUP BY aq.agent_qc_id, us.cognito_username, ag.agent_name
        """
        records = await self.pg.generic_read(sql, {"user_id": user_id})
        return [AgentQC(**record) for record in records]

    async def search_agent_qc(self, criteria: List[HorizonCriteria]) -> List[AgentQC]:
        sql = """
        SELECT
            aqc.agent_qc_id::TEXT, aqc.agent_id::TEXT, ag.agent_name, aqc.plan_id::TEXT, aqc.user_id::TEXT,
            aqc.query, aqc.agent_status,
            aqc.cs_reviewer::TEXT, aqc.eng_reviewer::TEXT, aqc.prod_reviewer::TEXT, aqc.follow_up, aqc.score_rating,
            aqc.priority, aqc.use_case, aqc.problem_area, aqc.cs_failed_reason, aqc.cs_attempt_reprompting,
            aqc.cs_expected_output, aqc.cs_notes, aqc.canned_prompt_id, aqc.eng_failed_reason, aqc.eng_solution,
            aqc.eng_solution_difficulty, aqc.jira_link, aqc.slack_link, aqc.fullstory_link, aqc.duplicate_agent::TEXT,
            aqc.created_at, aqc.last_updated,
            cw.cloudwatch_url
        FROM agent.agent_qc aqc
        LEFT JOIN agent.agents ag ON aqc.agent_id = ag.agent_id
        LEFT JOIN
            (SELECT all_tl.agent_id, all_tl.cloudwatch_url
                FROM (SELECT tl.agent_id, MAX(tl.finished_at) as latest
                        FROM boosted_dag.task_log tl
                        WHERE tl.agent_id IS NOT NULL AND tl.agent_id != '' AND tl.cloudwatch_url IS NOT NULL
                        GROUP BY tl.agent_id)
                AS latest_by_agent_id
                LEFT JOIN boosted_dag.task_log all_tl
                ON all_tl.agent_id = latest_by_agent_id.agent_id and all_tl.finished_at = latest_by_agent_id.latest)
                AS cw ON cw.agent_id = aqc.agent_id::TEXT
        """
        search_params = {}
        # Dynamically add conditions based on HorizonCriteria
        for idx, criterion in enumerate(criteria):
            if idx == 0:
                sql += "\nWHERE"
            else:
                sql += " AND"
            param1 = f"arg1_{idx}"
            param2 = f"arg2_{idx}"
            if criterion.operator == "BETWEEN":
                sql += f" {criterion.column} BETWEEN %({param1})s AND %({param2})s"
                search_params[param1] = criterion.arg1
                search_params[param2] = criterion.arg2
            elif criterion.operator == "ILIKE":
                sql += f" {criterion.column} ILIKE %({param1})s"
                search_params[param1] = f"%{criterion.arg1}%"  # Add wildcards for partial matches
            elif criterion.operator == "=":
                sql += f" {criterion.column} = %({param1})s"
                search_params[param1] = criterion.arg1
            elif criterion.operator == "IN":
                sql += f" {criterion.column} IN %({param1})s"
                search_params[param1] = tuple(
                    criterion.arg1
                )  # Ensure arg1 is a tuple for IN queries
            elif criterion.operator == ">":
                sql += f" {criterion.column} > %({param1})s"
                search_params[param1] = criterion.arg1
            elif criterion.operator == "<":
                sql += f" {criterion.column} < %({param1})s"
                search_params[param1] = criterion.arg1
            elif criterion.operator == "!=":
                sql += f" {criterion.column} != %({param1})s"
                search_params[param1] = criterion.arg1
            else:
                logging.warning(f"Unknown operator {criterion.operator=}")

        # Always sort by latest agent ran
        sql += " ORDER BY aqc.created_at DESC"
        records = await self.pg.generic_read(sql, search_params)
        return [AgentQC(**record) for record in records]


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
