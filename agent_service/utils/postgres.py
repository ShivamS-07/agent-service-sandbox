from functools import lru_cache
from typing import List, Optional

from gbi_common_py_utils.utils.postgres import PostgresBase

from agent_service.endpoints.models import AgentMetadata
from agent_service.io_type_utils import IOType, dump_io_type
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.environment import EnvironmentUtils

PSQL_CONN = None


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
    def create_agent_for_user(self, user_id: str, agent_name: str = "New Chat") -> str:
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

    def delete_agent_by_id(self, agent_id: str) -> None:
        """
        This function deletes an agent.

        Args:
            agent_id: The agent id to delete.
        """
        sql = """
            DELETE FROM agent.agents
            WHERE agent_id = %(agent_id)s;
        """
        self.generic_write(sql, params={"agent_id": agent_id})

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

    def get_chat(self, agent_id: str) -> ChatContext:
        sql = """
        SELECT message AS content, message_time AS timestamp, is_user_message AS is_user
        FROM agent.chat_messages
        WHERE agent_id = %(agent_id)s
        ORDER BY timestamp
        """
        rows = self.generic_read(sql, {"agent_id": agent_id})
        return ChatContext(messages=[Message(**row) for row in rows])

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
