import datetime
from typing import Any, Dict, List, Optional, Tuple

from agent_service.endpoints.models import AgentMetadata, AgentOutput
from agent_service.io_type_utils import ComplexIOBase, IOType, load_io_type

# Make sure all io_types are registered
from agent_service.io_types import *  # noqa
from agent_service.io_types.output import Output
from agent_service.io_types.text import Text
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext, Message
from agent_service.utils.boosted_pg import BoostedPG, InsertToTableArgs
from agent_service.utils.date_utils import get_now_utc


async def get_output_from_io_type(val: IOType, pg: BoostedPG) -> Output:
    if not isinstance(val, ComplexIOBase):
        val = Text.from_io_type(val)
    val = await val.to_rich_output(pg)
    return val


class AsyncDB:
    def __init__(self, pg: BoostedPG):
        self.pg = pg

    async def get_agent_outputs(self, agent_id: str) -> List[AgentOutput]:
        sql = """
                SELECT plan_id::VARCHAR, plan_run_id::VARCHAR, output_id::VARCHAR, is_intermediate,
                    "output", created_at
                FROM agent.agent_outputs ao
                WHERE plan_run_id IN (
                    SELECT plan_run_id FROM agent.agent_outputs
                    WHERE agent_id = %(agent_id)s AND "output" NOTNULL AND is_intermediate = FALSE
                    ORDER BY created_at DESC LIMIT 1
                )
                ORDER BY created_at ASC;
                """
        rows = await self.pg.generic_read(sql, {"agent_id": agent_id})
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

    async def get_log_data_from_log_id(self, agent_id: str, log_id: str) -> List[Dict]:
        # NOTE: the reason to not return the `log_data` directly is because we can't distinguish
        # 1) if there's no such entry in the table
        # 2) if the entry exists but the `log_data` is None
        # these two cases will be handled differently so just return `rows`
        sql = """
            SELECT log_data
            FROM agent.work_logs
            WHERE agent_id = %(agent_id)s AND log_id = %(log_id)s
        """
        rows = await self.pg.generic_read(sql, {"agent_id": agent_id, "log_id": log_id})
        return rows

    async def get_latest_execution_plan(
        self, agent_id: str
    ) -> Tuple[Optional[str], Optional[ExecutionPlan]]:
        sql = """
            SELECT plan_id::VARCHAR, plan
            FROM agent.execution_plans
            WHERE agent_id = %(agent_id)s
            ORDER BY last_updated DESC
            LIMIT 1;
        """
        rows = await self.pg.generic_read(sql, params={"agent_id": agent_id})
        if not rows:
            return None, None
        return rows[0]["plan_id"], ExecutionPlan.model_validate(rows[0]["plan"])

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
            SELECT plan_id::VARCHAR, plan_run_id::VARCHAR, task_id::VARCHAR, is_task_output,
                log_id::VARCHAR, log_message, created_at
            FROM agent.work_logs
            WHERE agent_id = %(agent_id)s {filters}
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
            dt_filter += " AND message_time >= %(start)s"
            params["start"] = start
        if end:
            dt_filter += " AND message_time <= %(end)s"
            params["end"] = end

        sql = f"""
            SELECT message_id::VARCHAR, message, is_user_message, message_time
            FROM agent.chat_messages
            WHERE agent_id = %(agent_id)s{dt_filter}
            ORDER BY message_time ASC;
        """
        rows = await self.pg.generic_read(sql, params=params)
        return ChatContext(messages=[Message(agent_id=agent_id, **row) for row in rows])

    async def insert_agent_and_messages(
        self, agent_metadata: AgentMetadata, messages: List[Message]
    ) -> None:
        await self.pg.insert_atomic(
            to_insert=[
                InsertToTableArgs(table_name="agent.agents", rows=[agent_metadata.model_dump()]),
                InsertToTableArgs(
                    table_name="agent.chat_messages", rows=[msg.model_dump() for msg in messages]
                ),
            ]
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
            SELECT agent_id::VARCHAR, user_id::VARCHAR, agent_name, created_at, last_updated
            FROM agent.agents
            WHERE user_id = %(user_id)s;
        """
        rows = await self.pg.generic_read(sql, params={"user_id": user_id})
        return [
            AgentMetadata(
                agent_id=row["agent_id"],
                user_id=row["user_id"],
                agent_name=row["agent_name"],
                created_at=row["created_at"],
                last_updated=row["last_updated"],
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
            table_name="agent.chat_messages", rows=[msg.model_dump() for msg in messages]
        )
