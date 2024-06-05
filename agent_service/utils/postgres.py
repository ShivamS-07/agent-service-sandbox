import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from gbi_common_py_utils.utils.postgres import PostgresBase

from agent_service.endpoints.models import AgentMetadata
from agent_service.external.sec_utils import SecurityMetadata
from agent_service.io_type_utils import IOType, dump_io_type, load_io_type
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG, InsertToTableArgs
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.environment import EnvironmentUtils

PSQL_CONN = None
PSQL_CONN_SKIP_COMMIT = None

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

    def insert_agent(self, agent_metadata: AgentMetadata) -> None:
        self.multi_row_insert(table_name="agent.agents", rows=[agent_metadata.model_dump()])

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
    ) -> Tuple[Optional[str], Optional[ExecutionPlan], Optional[datetime.datetime]]:
        sql = """
            SELECT plan_id::VARCHAR, plan, created_at
            FROM agent.execution_plans
            WHERE agent_id = %(agent_id)s
            ORDER BY last_updated DESC
            LIMIT 1;
        """
        rows = self.generic_read(sql, params={"agent_id": agent_id})
        if not rows:
            return None, None, None
        row = rows[0]
        return row["plan_id"], ExecutionPlan.model_validate(row["plan"]), row["created_at"]

    def get_all_execution_plans(
        self, agent_id: str
    ) -> Tuple[List[ExecutionPlan], List[datetime.datetime]]:
        sql = """
            SELECT plan, created_at
            FROM agent.execution_plans
            WHERE agent_id = %(agent_id)s
            ORDER BY last_updated ASC
        """
        rows = self.generic_read(sql, params={"agent_id": agent_id})
        return [ExecutionPlan.model_validate(row["plan"]) for row in rows], [
            row["created_at"] for row in rows
        ]

    def get_agent_plan_runs(self, agent_id: str, limit_num: Optional[int] = None) -> List[str]:
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
        rows = self.generic_read(sql, params=params)

        return [row["plan_run_id"] for row in rows]

    def insert_plan_run(self, agent_id: str, plan_id: str, plan_run_id: str) -> None:
        self.insert_into_table(
            table_name="agent.plan_runs",
            agent_id=agent_id,
            plan_id=plan_id,
            plan_run_id=plan_run_id,
        )

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
            SELECT plan_id::VARCHAR, plan_run_id::VARCHAR, task_id::VARCHAR, is_task_output,
                log_id::VARCHAR, log_message, created_at
            FROM agent.work_logs
            WHERE agent_id = %(agent_id)s {filters}
            ORDER BY created_at DESC;
        """
        return self.generic_read(sql1, params=params)

    ################################################################################################
    # Tools and Execution Plans
    ################################################################################################
    def write_execution_plan(self, plan_id: str, agent_id: str, plan: ExecutionPlan) -> None:
        sql = """
        INSERT INTO agent.execution_plans (plan_id, agent_id, plan, created_at, last_updated)
        VALUES (
          %(plan_id)s, %(agent_id)s, %(plan)s, %(created_at)s, %(last_updated)s
        )
        ON CONFLICT (plan_id) DO UPDATE SET
          agent_id = EXCLUDED.agent_id,
          plan = EXCLUDED.plan,
          last_updated = NOW()
        """

        created_at = last_updated = get_now_utc()  # need so skip_commit db has proper times
        self.generic_write(
            sql,
            params={
                "plan_id": plan_id,
                "agent_id": agent_id,
                "plan": plan.model_dump_json(),
                "created_at": created_at,
                "last_updated": last_updated,
            },
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
          (agent_id, plan_id, plan_run_id, task_id, log_data, is_task_output)
        VALUES
          (
             %(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(task_id)s, %(log_data)s, TRUE
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

    def write_agent_output(
        self, output: IOType, context: PlanRunContext, is_intermediate: bool = False
    ) -> None:
        sql = """
        INSERT INTO agent.agent_outputs
          (agent_id, plan_id, plan_run_id, output, is_intermediate)
        VALUES
          (
             %(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(output)s, %(is_intermediate)s
          )
        """
        self.generic_write(
            sql,
            params={
                "agent_id": context.agent_id,
                "plan_id": context.plan_id,
                "plan_run_id": context.plan_run_id,
                "output": dump_io_type(output),
                "is_intermediate": is_intermediate,
            },
        )

    def get_sec_metadata_from_gbi(self, gbi_ids: List[int]) -> Dict[int, SecurityMetadata]:
        sql = """
            SELECT gbi_security_id AS gbi_id, isin, symbol AS ticker, name AS company_name,
                currency, security_region
            FROM master_security
            WHERE gbi_security_id = ANY(%s)
        """
        records = self.generic_read(sql, params=[gbi_ids])
        return {record["gbi_id"]: SecurityMetadata(**record) for record in records}

    # TODO this won't be needed once we merge the sync and async files, but for
    # now will just keep this here.
    def get_latest_agent_output(self, agent_id: str) -> Optional[IOType]:
        sql = """
        SELECT "output"
        FROM agent.agent_outputs ao
        WHERE plan_run_id IN (
            SELECT plan_run_id FROM agent.agent_outputs
            WHERE agent_id = %(agent_id)s AND "output" NOTNULL AND is_intermediate = FALSE
            ORDER BY created_at DESC LIMIT 1
        )
        ORDER BY created_at ASC
        LIMIT 1;
        """
        rows = self.generic_read(sql, {"agent_id": agent_id})
        if not rows:
            return None

        row = rows[0]
        output = load_io_type(row["output"])
        return output

    def get_short_company_description(
        self, gbi_id: int
    ) -> Tuple[Optional[str], Optional[datetime.datetime]]:
        """
        Given a GBI ID, return the short company description if present, otherwise None.
        """
        sql = """
        SELECT
          cds.company_description_short AS company_description,
          cds.last_updated AT TIME ZONE 'UTC' AS last_updated
        FROM spiq_security_mapping ssm
        LEFT JOIN nlp_service.company_descriptions_short cds
          ON ssm.spiq_company_id = cds.spiq_company_id
        WHERE ssm.gbi_id = %s;
        """
        records = self.generic_read(sql, params=[gbi_id])
        if not records:
            (None, None)
        row = records[0]
        return (row["company_description"], row["last_updated"])

    def get_long_company_description(
        self, gbi_id: int
    ) -> Tuple[Optional[str], Optional[datetime.datetime]]:
        """
        Given a GBI ID, return the long company description and updated time if
        present, otherwise None.
        """
        sql = """
        SELECT
          cd.company_description,
          cd.last_updated AT TIME ZONE 'UTC' AS last_updated
        FROM spiq_security_mapping ssm
        LEFT JOIN nlp_service.company_descriptions cd
          ON ssm.spiq_company_id = cd.spiq_company_id
        WHERE ssm.gbi_id = %s;
        """
        records = self.generic_read(sql, params=[gbi_id])
        if not records:
            (None, None)
        row = records[0]
        return (row["company_description"], row["last_updated"])


def get_psql(skip_commit: bool = False) -> Postgres:
    """
    This method fetches the global Postgres connection, so we only need one.
    If it does not exist, then initialize the connection.

    Returns: Postgres object, which inherits from PostgresBase.
    """
    global PSQL_CONN, PSQL_CONN_SKIP_COMMIT
    if skip_commit:
        if PSQL_CONN_SKIP_COMMIT is None:
            PSQL_CONN_SKIP_COMMIT = Postgres(skip_commit=skip_commit)
        return PSQL_CONN_SKIP_COMMIT

    else:
        if PSQL_CONN is None:
            PSQL_CONN = Postgres(skip_commit=skip_commit)
        return PSQL_CONN


# TODO eventually we can get rid of this possibly? Essentially allows us to use
# the AsyncDB functions anywhere. These will simply be blocking versions.
class SyncBoostedPG(BoostedPG):
    def __init__(self) -> None:
        self.db = get_psql()

    async def generic_read(self, sql: str, params: Optional[Any] = None) -> List[Dict[str, Any]]:
        return self.db.generic_read(sql, params)

    async def generic_write(self, sql: str, params: Optional[Any] = None) -> None:
        self.db.generic_write(sql, params)

    async def delete_from_table_where(self, table_name: str, **kwargs: Any) -> None:
        self.db.delete_from_table_where(table_name=table_name, **kwargs)

    async def generic_update(self, table_name: str, where: Dict, values_to_update: Dict) -> None:
        self.db.generic_update(
            table_name=table_name, where=where, values_to_update=values_to_update
        )

    async def multi_row_insert(
        self, table_name: str, rows: List[Dict[str, Any]], ignore_conflicts: bool = False
    ) -> None:
        self.db.multi_row_insert(
            table_name=table_name, rows=rows, ignore_conflicts=ignore_conflicts
        )

    async def insert_atomic(self, to_insert: List[InsertToTableArgs]) -> None:
        with self.db.transaction_cursor() as cursor:
            for arg in to_insert:
                sql, params = self.db._gen_multi_row_insert(
                    table_name=arg.table_name, values_to_insert=arg.rows, ignore_conficts=False
                )
                cursor.execute(sql, params)  # type: ignore
