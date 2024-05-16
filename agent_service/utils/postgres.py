import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from gbi_common_py_utils.utils.postgres import PostgresBase

from agent_service.endpoints.models import AgentMetadata
from agent_service.io_type_utils import IOType, dump_io_type
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.environment import EnvironmentUtils

PSQL_CONN = None

DEFAULT_AGENT_NAME = "New Chat"


class Postgres(PostgresBase):
    """
    This class is a wrapper over gbi-common-py-utils PostgresBase class.
    Please only add **reuseable** methods here.
    Otherwise please create specific db functions under each working directory
    """

    def __init__(self, skip_commit: bool = False, environment: Optional[str] = None):
        environment: str = environment or EnvironmentUtils.aws_ssm_prefix
        super().__init__(environment, skip_commit=skip_commit)
        self._env = environment

    ################################################################################################
    # Agent Service
    ################################################################################################
    def create_agent_for_user(self, user_id: str, agent_name: str = DEFAULT_AGENT_NAME) -> str:
        """
        This function creates an agent for a given user.

        Args:
            user_id: The user id to create the agent for.
            agent_name: The name of the agent.

        Returns: The agent id that was created.
        """
        sql = """
            INSERT INTO agent.agents (user_id, agent_name)
            VALUES (%(user_id)s, %(agent_name)s)
            RETURNING agent_id::VARCHAR;
        """
        rows = self.generic_read(sql, params={"user_id": user_id, "agent_name": agent_name})
        return rows[0]["agent_id"]

    def insert_agent_and_messages(
        self, agent_metadata: AgentMetadata, messages: List[Message]
    ) -> None:
        sql1 = """
            INSERT INTO agent.agents (agent_id, user_id, agent_name, created_at, last_updated)
            VALUES (%(agent_id)s, %(user_id)s, %(agent_name)s, %(created_at)s, %(last_updated)s)
        """
        sql2 = """
            INSERT INTO agent.chat_messages
            (agent_id, message_id, message, is_user_message, message_time)
            VALUES (%(agent_id)s, %(message_id)s, %(message)s, %(is_user_message)s,
                %(message_time)s)
        """
        with self.transaction_cursor() as cursor:
            cursor.execute(sql1, agent_metadata.model_dump())
            cursor.executemany(sql2, [msg.model_dump() for msg in messages])

    def delete_agent_by_id(self, agent_id: str) -> None:
        self.delete_from_table_where(table_name="agent.agents", agent_id=agent_id)

    def update_agent_name(self, agent_id: str, agent_name: str) -> None:
        return self.generic_update(
            table_name="agent.agents",
            where={"agent_id": agent_id},
            values_to_update={"agent_name": agent_name, "last_updated": get_now_utc()},
        )

    def get_user_all_agents(self, user_id: str) -> List[AgentMetadata]:
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
        rows = self.generic_read(sql, params={"user_id": user_id})
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

    @lru_cache(maxsize=128)
    def get_agent_owner(self, agent_id: str) -> Optional[str]:
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
        rows = self.generic_read(sql, params={"agent_id": agent_id})
        return rows[0]["user_id"] if rows else None

    def insert_chat_messages(self, messages: List[Message]) -> None:
        self.multi_row_insert(
            table_name="agent.chat_messages", rows=[msg.model_dump() for msg in messages]
        )

    def get_chats_history_for_agent(
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
        rows = self.generic_read(sql, params=params)
        return ChatContext(messages=[Message(agent_id=agent_id, **row) for row in rows])

    def get_latest_execution_plan(
        self, agent_id: str
    ) -> Tuple[Optional[str], Optional[ExecutionPlan]]:
        sql = """
            SELECT plan_id, plan
            FROM agent.execution_plans
            WHERE agent_id = %(agent_id)s
            ORDER BY last_updated DESC
            LIMIT 1;
        """
        rows = self.generic_read(sql, params={"agent_id": agent_id})
        if not rows:
            return None, None
        return rows[0]["plan_id"], ExecutionPlan.model_validate(rows[0]["plan"])

    def get_agent_plan_runs(self, agent_id: str, limit_num: Optional[int] = None) -> List[str]:
        if not limit_num:
            sql = """
                SELECT DISTINCT plan_run_id::VARCHAR
                FROM agent.work_logs
                WHERE agent_id = %(agent_id)s
            """
            rows = self.generic_read(sql, params={"agent_id": agent_id})
        else:
            sql = """
                WITH t AS (
                    SELECT plan_run_id::VARCHAR, MAX(created_at) AS created_at
                    FROM agent.work_logs wl
                    WHERE agent_id = %(agent_id)s
                    GROUP BY plan_run_id
                )
                SELECT plan_run_id
                FROM t
                ORDER BY created_at DESC
                LIMIT %(limit_num)s
            """
            rows = self.generic_read(sql, params={"agent_id": agent_id, "limit_num": limit_num})

        return [row["plan_run_id"] for row in rows]

    def get_agent_worklogs(
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
            SELECT plan_id::VARCHAR, plan_run_id::VARCHAR, task_id::VARCHAR, log_id::VARCHAR,
                log_message, created_at
            FROM agent.work_logs
            WHERE agent_id = %(agent_id)s AND is_task_output IS FALSE {filters}
            ORDER BY created_at DESC;
        """
        return get_psql().generic_read(sql1, params=params)

    def get_log_data_from_log_id(self, agent_id: str, log_id: str) -> List[Dict]:
        # NOTE: the reason to not return the `log_data` directly is because we can't distinguish
        # 1) if there's no such entry in the table
        # 2) if the entry exists but the `log_data` is None
        # these two cases will be handled differently so just return `rows`
        sql = """
            SELECT log_data
            FROM agent.work_logs
            WHERE agent_id = %(agent_id)s AND log_id = %(log_id)s
        """
        rows = self.generic_read(sql, {"agent_id": agent_id, "log_id": log_id})
        return rows

    ################################################################################################
    # Tools and Execution Plans
    ################################################################################################
    def write_execution_plan(self, plan_id: str, agent_id: str, plan: ExecutionPlan) -> None:
        sql = """
        INSERT INTO agent.execution_plans (plan_id, agent_id, plan)
        VALUES (
          %(plan_id)s, %(agent_id)s, %(plan)s
        )
        ON CONFLICT (plan_id) DO UPDATE SET
          agent_id = EXCLUDED.agent_id,
          plan = EXCLUDED.plan,
          last_updated = NOW()
        """
        self.generic_write(
            sql, params={"plan_id": plan_id, "agent_id": agent_id, "plan": plan.model_dump_json()}
        )

    def get_execution_plans(self, plan_ids: List[str]) -> Dict[str, ExecutionPlan]:
        sql = """
            SELECT plan_id::VARCHAR, plan
            FROM agent.execution_plans
            WHERE plan_id = ANY(%(plan_ids)s)
        """
        rows = self.generic_read(sql, params={"plan_ids": plan_ids})
        return {row["plan_id"]: ExecutionPlan.model_validate(row["plan"]) for row in rows}

    def write_tool_log(
        self, log: IOType, context: PlanRunContext, associated_data: Optional[IOType] = None
    ) -> None:
        sql = """
        INSERT INTO agent.work_logs
          (agent_id, plan_id, plan_run_id, task_id, log_message, log_data)
        VALUES
          (
             %(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(task_id)s,
             %(log_message)s, %(log_data)s
          )
        """
        self.generic_write(
            sql,
            params={
                "agent_id": context.agent_id,
                "plan_id": context.plan_id,
                "plan_run_id": context.plan_run_id,
                "task_id": context.task_id,
                "log_message": dump_io_type(log),
                "log_data": dump_io_type(associated_data) if associated_data else None,
            },
        )

    def write_tool_output(self, output: IOType, context: PlanRunContext) -> None:
        sql = """
        INSERT INTO agent.work_logs
          (agent_id, plan_id, plan_run_id, task_id, log_data)
        VALUES
          (
             %(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(task_id)s, %(log_data)s
          )
        """
        self.generic_write(
            sql,
            params={
                "agent_id": context.agent_id,
                "plan_id": context.plan_id,
                "plan_run_id": context.plan_run_id,
                "task_id": context.task_id,
                "log_data": dump_io_type(output),
            },
        )


def get_psql(skip_commit: bool = False) -> Postgres:
    """
    This method fetches the global Postgres connection, so we only need one.
    If it does not exist, then initialize the connection.

    Returns: Postgres object, which inherits from PostgresBase.
    """
    global PSQL_CONN
    if PSQL_CONN is None:
        PSQL_CONN = Postgres(skip_commit=skip_commit)
    elif not PSQL_CONN.skip_commit and skip_commit:
        PSQL_CONN = Postgres(skip_commit=skip_commit)
    return PSQL_CONN
