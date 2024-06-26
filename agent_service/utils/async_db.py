import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from agent_service.endpoints.models import AgentMetadata, AgentOutput
from agent_service.io_type_utils import IOType, load_io_type

# Make sure all io_types are registered
from agent_service.io_types import *  # noqa
from agent_service.planner.planner_types import ExecutionPlan, PlanStatus
from agent_service.types import ChatContext, Message, Notification
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.output_utils.output_construction import get_output_from_io_type
from agent_service.utils.postgres import Postgres


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
                AND plan_id = %(plan_id)s
                AND plan_run_id != %(latest_plan_run_id)s
        GROUP BY plan_run_id, created_at
        ORDER BY plan_run_id, created_at DESC
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

    async def get_agent_outputs(
        self, agent_id: str, plan_run_id: Optional[str] = None
    ) -> List[AgentOutput]:
        where_clause = """
            ao.plan_run_id IN (
                    SELECT plan_run_id FROM agent.agent_outputs
                    WHERE agent_id = %(agent_id)s AND "output" NOTNULL AND is_intermediate = FALSE
                    ORDER BY created_at DESC LIMIT 1
                )
        """

        params: Dict[str, Any] = {"agent_id": agent_id}
        if plan_run_id:
            where_clause = "ao.plan_run_id = %(plan_run_id)s AND ao.output NOTNULL AND ao.is_intermediate = false"
            params["plan_run_id"] = plan_run_id

        sql = f"""
                SELECT ao.plan_id::VARCHAR, ao.plan_run_id::VARCHAR, ao.output_id::VARCHAR, ao.is_intermediate,
                    ao.output, ao.created_at, pr.shared
                FROM agent.agent_outputs ao
                LEFT JOIN agent.plan_runs pr
                ON ao.plan_run_id = pr.plan_run_id
                WHERE {where_clause}
                ORDER BY created_at ASC;
                """
        rows = await self.pg.generic_read(sql, params)
        if not rows:
            return []

        outputs = []
        for row in rows:
            output = row["output"]
            output_value = load_io_type(output) if output else output
            output_value = await get_output_from_io_type(output_value, pg=self.pg)
            row["output"] = output_value
            outputs.append(AgentOutput(agent_id=agent_id, **row))

        return outputs

    async def is_cancelled(self, ids_to_check: List[str]) -> bool:
        """
        Returns true if ANY of the input ID's have been cancelled.
        """
        sql = """
        select * from agent.cancelled_ids where cancelled_id = ANY(%(ids_to_check)s)
        """
        rows = await self.pg.generic_read(sql, {"ids_to_check": ids_to_check})
        return len(rows) > 0

    async def get_plan_run_outputs(self, plan_run_id: str) -> List[AgentOutput]:
        sql = """
                SELECT ao.agent_id::VARCHAR, ao.plan_id::VARCHAR, ao.plan_run_id::VARCHAR,
                  ao.output_id::VARCHAR, ao.is_intermediate,
                    ao.output, ao.created_at, pr.shared
                FROM agent.agent_outputs ao
                LEFT JOIN agent.plan_runs pr
                ON ao.plan_run_id = pr.plan_run_id
                WHERE ao.plan_run_id = %(plan_run_id)s
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
            outputs.append(AgentOutput(**row))

        return outputs

    async def get_task_output(
        self, agent_id: str, plan_run_id: str, task_id: str
    ) -> Optional[IOType]:
        sql = """
        SELECT log_data
        FROM agent.work_logs
        WHERE agent_id = %(agent_id)s AND plan_run_id = %(plan_run_id)s AND task_id = %(task_id)s
            AND is_task_output AND log_data NOTNULL
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
            AND NOT is_task_output AND log_data NOTNULL
        ORDER BY created_at DESC
        LIMIT 1;
        """
        rows = await self.pg.generic_read(
            sql, {"agent_id": agent_id, "plan_run_id": plan_run_id, "log_id": log_id}
        )
        if not rows:
            return None
        return load_io_type(rows[0]["log_data"])

    async def get_latest_execution_plan(
        self, agent_id: str
    ) -> Tuple[Optional[str], Optional[ExecutionPlan], Optional[datetime.datetime], Optional[str]]:
        sql = """
            SELECT plan_id::VARCHAR, plan, created_at, status
            FROM agent.execution_plans
            WHERE agent_id = %(agent_id)s
            ORDER BY last_updated DESC
            LIMIT 1;
        """
        rows = await self.pg.generic_read(sql, params={"agent_id": agent_id})
        if not rows:
            return None, None, None, None
        return (
            rows[0]["plan_id"],
            ExecutionPlan.model_validate(rows[0]["plan"]),
            rows[0]["created_at"],
            rows[0]["status"],
        )

    @lru_cache(maxsize=128)
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

    async def get_agent_plan_runs(
        self, agent_id: str, limit_num: Optional[int] = None
    ) -> List[str]:
        limit_sql = ""
        params: Dict[str, Any] = {"agent_id": agent_id}
        if limit_num:
            limit_sql = "LIMIT %(limit_num)s"
            params["limit_num"] = limit_num

        sql = f"""
        SELECT plan_run_id::VARCHAR FROM agent.plan_runs
        WHERE agent_id = %(agent_id)s
        ORDER BY created_at DESC
        {limit_sql}
        """
        rows = await self.pg.generic_read(sql, params=params)

        return [row["plan_run_id"] for row in rows]

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
            filters += " AND created_at >= %(start_date)s"
            params["start_date"] = start_date
        if end_date:
            filters += " AND created_at < %(end_date)s"
            params["end_date"] = end_date
        if plan_run_ids:
            filters += " AND plan_run_id = ANY(%(plan_run_ids)s)"
            params["plan_run_ids"] = plan_run_ids

        sql1 = f"""
            SELECT wl.plan_id::VARCHAR, wl.plan_run_id::VARCHAR, wl.task_id::VARCHAR, wl.is_task_output,
                wl.log_id::VARCHAR, wl.log_message, wl.created_at, pr.shared, (log_data NOTNULL) AS has_output
            FROM agent.work_logs wl
            LEFT JOIN agent.plan_runs pr
            ON wl.plan_run_id = pr.plan_run_id
            WHERE wl.agent_id = %(agent_id)s {filters}
            ORDER BY created_at DESC;
        """
        return await self.pg.generic_read(sql1, params=params)

    async def get_execution_plans(self, plan_ids: List[str]) -> Dict[str, ExecutionPlan]:
        sql = """
            SELECT plan_id::VARCHAR, plan
            FROM agent.execution_plans
            WHERE plan_id = ANY(%(plan_ids)s)
        """
        rows = await self.pg.generic_read(sql, params={"plan_ids": plan_ids})
        return {row["plan_id"]: ExecutionPlan.model_validate(row["plan"]) for row in rows}

    async def get_chats_history_for_agent(
        self,
        agent_id: str,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
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

        sql = f"""
            SELECT cm.message_id::VARCHAR, cm.message, cm.is_user_message, cm.message_time,
              COALESCE(nf.unread, FALSE) as unread
            FROM agent.chat_messages cm
            LEFT JOIN agent.notifications nf
            ON cm.message_id = nf.message_id
            WHERE cm.agent_id = %(agent_id)s{dt_filter}
            ORDER BY cm.message_time ASC;
        """
        rows = await self.pg.generic_read(sql, params=params)

        return ChatContext(messages=[Message(agent_id=agent_id, **row) for row in rows])

    async def create_agent(self, agent_metadata: AgentMetadata) -> None:
        await self.pg.multi_row_insert(
            table_name="agent.agents", rows=[agent_metadata.to_agent_row()]
        )

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
        INSERT INTO agent.execution_plans (plan_id, agent_id, plan, created_at, last_updated, status)
        VALUES (
          %(plan_id)s, %(agent_id)s, %(plan)s, %(created_at)s, %(last_updated)s, %(status)s
        )
        ON CONFLICT (plan_id) DO UPDATE SET
          agent_id = EXCLUDED.agent_id,
          plan = EXCLUDED.plan,
          last_updated = NOW(),
          status = EXCLUDED.status
        """
        created_at = last_updated = get_now_utc()  # need so skip_commit db has proper times
        await self.pg.generic_write(
            sql,
            params={
                "plan_id": plan_id,
                "agent_id": agent_id,
                "plan": plan.model_dump_json(),
                "created_at": created_at,
                "last_updated": last_updated,
                "status": status.value,
            },
        )

    async def delete_agent_by_id(self, agent_id: str) -> None:
        await self.pg.delete_from_table_where(table_name="agent.agents", agent_id=agent_id)

    async def get_user_all_agents(self, user_id: str) -> List[AgentMetadata]:
        """
        This function retrieves all agents for a given user.

        Args:
            user_id: The user id to retrieve agents for.

        Returns: A list of all agents for the user.
        """
        sql = """
        SELECT DISTINCT ON (a.agent_id)
            a.agent_id::VARCHAR, a.user_id::VARCHAR, agent_name, a.created_at,
            a.last_updated, a.automation_enabled, pr.created_at AS last_run,
            n.summary AS latest_notification_string,
            (
              SELECT COUNT(*) FROM agent.notifications n2
              WHERE n2.agent_id = a.agent_id AND n2.unread
              GROUP BY n2.agent_id
            ) AS unread_notification_count
        FROM agent.agents a
        LEFT JOIN agent.plan_runs pr ON a.agent_id = pr.agent_id
        LEFT JOIN agent.notifications n ON a.agent_id = n.agent_id
        WHERE user_id = %(user_id)s
        ORDER BY a.agent_id, pr.created_at DESC, n.created_at DESC;
        """
        rows = await self.pg.generic_read(sql, params={"user_id": user_id})
        tomorrow = get_now_utc() + datetime.timedelta(days=1)
        return [
            AgentMetadata(
                agent_id=row["agent_id"],
                user_id=row["user_id"],
                agent_name=row["agent_name"],
                created_at=row["created_at"],
                last_updated=row["last_updated"],
                last_run=row["last_run"],
                next_run=tomorrow,  # TEMPORARY
                latest_notification_string=row["latest_notification_string"],
                automation_enabled=row["automation_enabled"],
                unread_notification_count=row["unread_notification_count"] or 0,
            )
            for row in rows
        ]

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
        self, agent_id: str, timestamp: Optional[datetime.datetime]
    ) -> None:
        where_clause = "agent_id = %(agent_id)s"
        if timestamp is not None:
            where_clause += " AND created_at <= %(timestamp)s"

        sql = f"""
        UPDATE agent.notifications SET unread = FALSE WHERE {where_clause}
        """

        params = {"agent_id": agent_id, "timestamp": timestamp}

        await self.pg.generic_write(sql, params=params)

    async def set_agent_automation_enabled(self, agent_id: str, enabled: bool) -> None:
        await self.pg.generic_update(
            table_name="agent.agents",
            where={"agent_id": agent_id},
            values_to_update={"automation_enabled": enabled},
        )


async def get_chat_history_from_db(agent_id: str, db: Union[AsyncDB, Postgres]) -> ChatContext:
    if isinstance(db, Postgres):
        return db.get_chats_history_for_agent(agent_id)
    elif isinstance(db, AsyncDB):
        return await db.get_chats_history_for_agent(agent_id)


async def get_latest_execution_plan_from_db(
    agent_id: str, db: Union[AsyncDB, Postgres]
) -> Tuple[Optional[str], Optional[ExecutionPlan], Optional[datetime.datetime], Optional[str]]:
    if isinstance(db, Postgres):
        return db.get_latest_execution_plan(agent_id)
    elif isinstance(db, AsyncDB):
        return await db.get_latest_execution_plan(agent_id)
    else:
        return None, None, None, None
