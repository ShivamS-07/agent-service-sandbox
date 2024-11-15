import datetime
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Tuple

from gbi_common_py_utils.utils.postgres import PostgresBase
from psycopg import Cursor

from agent_service.endpoints.models import AgentInfo, Status
from agent_service.io_type_utils import IOType, dump_io_type, load_io_type
from agent_service.planner.planner_types import (
    ExecutionPlan,
    OutputWithID,
    PlanStatus,
    SamplePlan,
)
from agent_service.types import ChatContext, Message, Notification, PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG, InsertToTableArgs
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.environment import EnvironmentUtils
from agent_service.utils.sec.sec_api import SecurityMetadata

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
            WHERE agent_id = %(agent_id)s AND NOT deleted;
        """
        rows = self.generic_read(sql, params={"agent_id": agent_id})
        return rows[0]["user_id"] if rows else None

    def insert_agent(self, agent_metadata: AgentInfo) -> None:
        self.multi_row_insert(table_name="agent.agents", rows=[agent_metadata.to_agent_row()])

    def insert_chat_messages(self, messages: List[Message]) -> None:
        self.multi_row_insert(
            table_name="agent.chat_messages", rows=[msg.to_message_row() for msg in messages]
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
            SELECT message_id::VARCHAR, message, is_user_message, message_time, message_author
            FROM agent.chat_messages
            WHERE agent_id = %(agent_id)s{dt_filter}
            ORDER BY message_time ASC;
        """
        rows = self.generic_read(sql, params=params)
        return ChatContext(messages=[Message(agent_id=agent_id, **row) for row in rows])

    def get_latest_execution_plan(self, agent_id: str) -> Tuple[
        Optional[str],
        Optional[ExecutionPlan],
        Optional[datetime.datetime],
        Optional[str],
        Optional[str],
    ]:
        sql = """
            SELECT ep.plan_id::VARCHAR, ep.plan, pr.created_at, ep.status,
              pr.plan_run_id::VARCHAR AS upcoming_plan_run_id
            FROM agent.execution_plans ep
            LEFT JOIN agent.plan_runs pr
            ON ep.plan_id = pr.plan_id
            WHERE ep.agent_id = %(agent_id)s
            ORDER BY ep.last_updated DESC
            LIMIT 1;
        """
        rows = self.generic_read(sql, params={"agent_id": agent_id})
        if not rows:
            return None, None, None, None, None
        row = rows[0]
        return (
            row["plan_id"],
            ExecutionPlan.from_dict(row["plan"]),
            row["created_at"],
            row["status"],
            row["upcoming_plan_run_id"],
        )

    def get_all_execution_plans(
        self, agent_id: str
    ) -> Tuple[List[ExecutionPlan], List[datetime.datetime], List[str]]:
        sql = """
            SELECT plan, created_at, plan_id::TEXT
            FROM agent.execution_plans ep
            WHERE agent_id = %(agent_id)s
            ORDER BY ep.last_updated ASC
        """
        rows = self.generic_read(sql, params={"agent_id": agent_id})
        return (
            [ExecutionPlan.from_dict(row["plan"]) for row in rows],
            [row["created_at"] for row in rows],
            [row["plan_id"] for row in rows],
        )

    def get_agent_plan_runs(
        self,
        agent_id: str,
        start_date: Optional[datetime.date] = None,  # inclusive
        end_date: Optional[datetime.date] = None,  # exclusive
        limit_num: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        params: Dict[str, Any] = {"agent_id": agent_id}

        limit_sql = ""
        if limit_num:
            limit_sql = "LIMIT %(limit_num)s"
            params["limit_num"] = limit_num

        start_date_filter = ""
        if start_date:
            params["start_date"] = start_date
            start_date_filter = " AND created_at >= %(start_date)s"

        end_date_filter = ""
        if end_date:
            params["end_date"] = end_date
            end_date_filter = " AND created_at < %(end_date)s"

        sql = f"""
        SELECT plan_run_id::VARCHAR, plan_id::VARCHAR
        FROM agent.plan_runs
        WHERE agent_id = %(agent_id)s{start_date_filter}{end_date_filter}
        ORDER BY created_at DESC
        {limit_sql}
        """
        rows = self.generic_read(sql, params=params)

        return [(row["plan_run_id"], row["plan_id"]) for row in rows]

    def get_plan_runs_for_plan_id(self, plan_id: str, agent_id: str) -> List[str]:
        params: Dict[str, Any] = {"plan_id": plan_id, "agent_id": agent_id}
        sql = """
        SELECT plan_run_id::VARCHAR FROM agent.plan_runs
        WHERE agent_id = %(agent_id)s
        AND plan_id = %(plan_id)s
        ORDER BY created_at DESC
        """
        rows = self.generic_read(sql, params=params)

        return [row["plan_run_id"] for row in rows]

    def get_plan_run(self, plan_run_id: str) -> Optional[Dict[str, Any]]:
        """
        Given a plan_run_id, return the plan run's info.
        """
        sql = """
        SELECT agent_id::VARCHAR, plan_id::VARCHAR, created_at, shared, status
        FROM agent.plan_runs WHERE plan_run_id = %(plan_run_id)s LIMIT 1
        """
        params: Dict[str, Any] = {"plan_run_id": plan_run_id}
        rows = self.generic_read(sql, params=params)

        if not rows:
            return None

        return rows[0]

    def get_running_plan_run(self, agent_id: str) -> Optional[Dict[str, str]]:
        """
        Look at `agent.plan_runs` table and find the latest plan run that is either NOT_STARTED or
        RUNNING. If there is no such plan run, return None.
        """

        sql = """
            SELECT plan_run_id::VARCHAR, plan_id::VARCHAR
            FROM agent.plan_runs
            WHERE agent_id = %(agent_id)s AND status = ANY(%(status)s)
            ORDER BY created_at DESC
            LIMIT 1
        """
        rows = self.generic_read(
            sql,
            params={
                "agent_id": agent_id,
                "status": [Status.NOT_STARTED.value, Status.RUNNING.value],
            },
        )
        return rows[0] if rows else None

    def insert_plan_run(self, agent_id: str, plan_id: str, plan_run_id: str) -> None:
        sql = """
        INSERT INTO agent.plan_runs (agent_id, plan_id, plan_run_id, created_at)
        VALUES (%(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(created_at)s)
        ON CONFLICT (plan_run_id) DO NOTHING
        """

        self.generic_write(
            sql,
            params={
                "agent_id": agent_id,
                "plan_id": plan_id,
                "plan_run_id": plan_run_id,
                "created_at": get_now_utc(),
            },
        )

    def cancel_agent_plan(
        self, plan_id: Optional[str] = None, plan_run_id: Optional[str] = None
    ) -> None:
        cancelled_ids = [{"cancelled_id": _id} for _id in (plan_id, plan_run_id) if _id]
        if not cancelled_ids:
            return

        self.multi_row_insert(table_name="agent.cancelled_ids", rows=cancelled_ids)

    def is_cancelled(self, ids_to_check: List[str]) -> bool:
        """
        Returns true if ANY of the input ID's have been cancelled.
        """
        sql = """
        select * from agent.cancelled_ids where cancelled_id = ANY(%(ids_to_check)s)
        """
        rows = self.generic_read(sql, {"ids_to_check": ids_to_check})
        return len(rows) > 0

    def is_agent_deleted(self, agent_id: Optional[str]) -> bool:
        if agent_id is None:
            return False

        sql = "SELECT deleted FROM agent.agents WHERE agent_id = %(agent_id)s"
        rows = self.generic_read(sql, {"agent_id": agent_id})
        if rows:
            return rows[0]["deleted"]
        return True

    def is_agent_draft(self, agent_id: Optional[str]) -> bool:
        if agent_id is None:
            return False

        sql = "SELECT is_draft FROM agent.agents WHERE agent_id = %(agent_id)s"
        rows = self.generic_read(sql, {"agent_id": agent_id})
        if rows:
            return rows[0]["is_draft"]
        return True

    def get_agent_worklogs(
        self,
        agent_id: str,
        start_date: Optional[datetime.date] = None,  # inclusive
        end_date: Optional[datetime.date] = None,  # exclusive
        plan_run_ids: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
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
        if task_ids:
            filters += " AND task_id = ANY(%(task_ids)s)"
            params["task_ids"] = task_ids

        sql1 = f"""
            SELECT plan_id::VARCHAR, plan_run_id::VARCHAR, task_id::VARCHAR, is_task_output,
                log_id::VARCHAR, log_message, created_at, (log_data NOTNULL) AS has_output
            FROM agent.work_logs
            WHERE agent_id = %(agent_id)s {filters}
            ORDER BY created_at DESC;
        """
        return self.generic_read(sql1, params=params)

    ################################################################################################
    # Tools and Execution Plans
    ################################################################################################
    def write_execution_plan(
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
          last_updated = %(last_updated)s,
          status = EXCLUDED.status
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
                "status": status.value,
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

    def write_tool_split_outputs(
        self, outputs_with_ids: List[OutputWithID], context: PlanRunContext
    ) -> None:
        now = get_now_utc()
        time_delta = datetime.timedelta(seconds=0.01)
        rows = [
            {
                "agent_id": context.agent_id,
                "plan_id": context.plan_id,
                "plan_run_id": context.plan_run_id,
                "log_id": output.output_id,
                "task_id": output.task_id,
                "log_data": dump_io_type(output.output),
                "is_task_output": True,
                "created_at": now + idx * time_delta,  # preserve order
            }
            for idx, output in enumerate(outputs_with_ids)
        ]
        self.multi_row_insert(table_name="agent.work_logs", rows=rows)

    def write_agent_output(
        self,
        output: IOType,
        output_id: str,
        context: PlanRunContext,
        is_intermediate: bool = False,
        live_plan_output: bool = False,
    ) -> None:
        sql = """
        INSERT INTO agent.agent_outputs
          (output_id, agent_id, plan_id, plan_run_id, output, is_intermediate, live_plan_output)
        VALUES
          (
             %(output_id)s, %(agent_id)s, %(plan_id)s, %(plan_run_id)s, %(output)s, %(is_intermediate)s,
             %(live_plan_output)s
          )
        """
        self.generic_write(
            sql,
            params={
                "output_id": output_id,
                "agent_id": context.agent_id,
                "plan_id": context.plan_id,
                "plan_run_id": context.plan_run_id,
                "output": dump_io_type(output),
                "is_intermediate": is_intermediate,
                "live_plan_output": live_plan_output,
            },
        )

    def write_agent_multi_outputs(
        self,
        outputs_with_ids: List[OutputWithID],
        context: PlanRunContext,
        is_intermediate: bool = False,
        live_plan_output: bool = False,
    ) -> None:
        now = get_now_utc()
        time_delta = datetime.timedelta(seconds=0.01)
        rows = [
            {
                "output_id": output_with_id.output_id,
                "agent_id": context.agent_id,
                "plan_id": context.plan_id,
                "plan_run_id": context.plan_run_id,
                "task_id": output_with_id.task_id,
                "output": dump_io_type(output_with_id.output),
                "is_intermediate": is_intermediate,
                "live_plan_output": live_plan_output,
                "created_at": now + idx * time_delta,  # preserve order
            }
            for idx, output_with_id in enumerate(outputs_with_ids)
        ]
        self.multi_row_insert(table_name="agent.agent_outputs", rows=rows)

    def get_sec_metadata_from_gbi(self, gbi_ids: List[int]) -> Dict[int, SecurityMetadata]:
        sql = """
            SELECT gbi_security_id AS gbi_id, isin, symbol AS ticker, name AS company_name,
                currency, security_region
            FROM master_security
            WHERE gbi_security_id = ANY(%s)
        """
        records = self.generic_read(sql, params=[gbi_ids])
        return {record["gbi_id"]: SecurityMetadata(**record) for record in records}

    def get_last_tool_output_for_plan(
        self, agent_id: str, plan_id: str, task_id: str
    ) -> Optional[IOType]:
        sql = """
        SELECT log_data
        FROM agent.work_logs wl
        WHERE agent_id = %(agent_id)s AND plan_id = %(plan_id)s AND task_id = %(task_id)s
          AND is_task_output
        ORDER BY created_at DESC
        LIMIT 1
        """
        rows = self.generic_read(
            sql, {"agent_id": agent_id, "plan_id": plan_id, "task_id": task_id}
        )
        if not rows:
            return None

        row = rows[0]
        output = load_io_type(row["log_data"])
        return output

    def get_short_company_descriptions_for_gbi_ids(
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
        records = self.generic_read(sql, params=[gbi_ids])
        return {r["gbi_id"]: (r["company_description"], r["last_updated"]) for r in records}

    def get_currency_exchange_to_usd(self, iso: str, date: datetime.date) -> Optional[float]:
        sql = """
        SELECT exchange_rate FROM data.currency_exchange
        WHERE iso = %(iso)s AND price_date = %(price_date)s
        """
        records = self.generic_read(sql, params={"iso": iso, "price_date": date})
        # We take the inverse to get the exchange rate from the iso -> usd
        return 1 / float(records[0].get("exchange_rate")) if records else None

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

    def get_company_descriptions(self, gbi_ids: List[int]) -> Dict[int, str]:
        sql = """
        SELECT DISTINCT ON (ssm.gbi_id) ssm.gbi_id, cds.company_description_short, cds.last_updated
        FROM spiq_security_mapping ssm
        JOIN nlp_service.company_descriptions_short cds
        ON cds.spiq_company_id = ssm.spiq_company_id
        WHERE ssm.gbi_id = ANY(%(gbi_ids)s)
        ORDER BY ssm.gbi_id, cds.last_updated DESC NULLS LAST;
        """

        # get short first since we always have that

        db = get_psql()
        rows = db.generic_read(sql, {"gbi_ids": gbi_ids})
        db.get_long_company_description
        descriptions_rows = {row["gbi_id"]: row["company_description_short"] for row in rows}
        descriptions = {gbi: descriptions_rows.get(gbi, "No description found") for gbi in gbi_ids}

        # replace with long if it exists

        long_sql = sql.replace("_short", "")

        rows = db.generic_read(long_sql, {"gbi_ids": gbi_ids})

        for row in rows:
            descriptions[row["gbi_id"]] = row["company_description"]
        return descriptions

    # Sample plan functions

    def insert_sample_plan(self, sample_plan: SamplePlan) -> None:
        self.insert_into_table(
            table_name="agent.sample_plans",
            sample_plan_id=sample_plan.id,
            input=sample_plan.input,
            plan=sample_plan.plan,
        )

    def get_all_sample_plans(self) -> List[SamplePlan]:
        sql = """
            SELECT sample_plan_id::TEXT, input, plan
            FROM agent.sample_plans
        """
        rows = self.generic_read(sql)
        return [
            SamplePlan(id=row["sample_plan_id"], input=row["input"], plan=row["plan"])
            for row in rows
        ]

    def delete_sample_plan(self, id: str) -> None:
        sql = "DELETE FROM agent.sample_plans WHERE sample_plan_id=%(id)s"
        self.generic_write(sql, {"id": id})

    def get_execution_plan_for_run(self, plan_run_id: str) -> ExecutionPlan:
        sql = """
            SELECT plan
            FROM agent.execution_plans ep
            JOIN agent.plan_runs pr ON ep.plan_id = pr.plan_id
            WHERE pr.plan_run_id = %(plan_run_id)s
        """
        rows = self.generic_read(sql, params={"plan_run_id": plan_run_id})
        return ExecutionPlan.from_dict(rows[0]["plan"])

    ################################################################################################
    # Notifications
    ################################################################################################

    def insert_notifications(self, notifications: List[Notification]) -> None:
        self.multi_row_insert(
            table_name="agent.notifications", rows=[notif.model_dump() for notif in notifications]
        )

    def get_notification_event_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
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

        rows = self.generic_read(sql, params={"agent_id": agent_id})
        if not rows:
            return None

        return rows[0]

    def insert_agent_custom_notification_prompt(
        self, agent_id: str, prompt: str, auto_generated: bool = True
    ) -> None:
        sql = """
        INSERT INTO agent.custom_notifications (agent_id, notification_prompt, created_at, auto_generated)
        VALUES (%(agent_id)s, %(prompt)s, %(created_at)s, %(auto_generated)s)
        """
        # need to manually insert created_at for offline tool
        self.generic_write(
            sql,
            {
                "agent_id": agent_id,
                "prompt": prompt,
                "created_at": get_now_utc(),
                "auto_generated": auto_generated,
            },
        )

    # Profile Generation
    def get_industry_names(self, gic_type: str = "GGROUP") -> List[str]:
        sql = """
            SELECT name, gictype
            FROM gic_sector
            WHERE gictype = (%s)
        """
        records = self.generic_read(sql, params=[gic_type])
        return [record["name"] for record in records]

    def get_scheduled_agents(self) -> List[str]:
        sql = "SELECT agent_id FROM agent.agents WHERE automation_enabled"
        rows = self.generic_read(sql)
        return [row["agent_id"] for row in rows]

    def get_agent_live_execution_plan(
        self, agent_id: str
    ) -> Tuple[Optional[str], Optional[ExecutionPlan]]:
        live_agents_info = self.get_live_agents_info(agent_ids=[agent_id])
        if live_agents_info:
            return live_agents_info[0]["plan_id"], ExecutionPlan.from_dict(
                live_agents_info[0]["plan"]
            )
        return None, None

    def get_live_agents_info(self, agent_ids: List[str]) -> List[Dict[str, Any]]:
        sql = """
        SELECT DISTINCT ON (ag.agent_id) ag.agent_id::TEXT, ag.user_id::TEXT, ep.plan_id::TEXT, ep.plan,
          ag.schedule, ep.created_at as plan_created_at
        FROM agent.agents ag JOIN agent.execution_plans ep ON ag.agent_id = ep.agent_id
        WHERE ag.agent_id = ANY(%(agent_ids)s) and ep.status = 'READY' and ep.automated_run
        ORDER BY ag.agent_id, ep.created_at DESC
        """
        rows = self.generic_read(sql, params={"agent_ids": agent_ids})
        if not rows:
            # If no plans marked for automated run, use latest plan
            sql = """
            SELECT DISTINCT ON (ag.agent_id) ag.agent_id::TEXT, ag.user_id::TEXT, ep.plan_id::TEXT, ep.plan,
                ag.schedule
            FROM agent.agents ag JOIN agent.execution_plans ep ON ag.agent_id = ep.agent_id
            WHERE ag.agent_id = ANY(%(agent_ids)s) and ep.status = 'READY'
            ORDER BY ag.agent_id, ep.created_at DESC
            """
            rows = self.generic_read(sql, params={"agent_ids": agent_ids})
        return rows

    def get_chat_contexts(self, agent_ids: List[str]) -> DefaultDict[str, ChatContext]:
        sql = """
            SELECT cm.agent_id::VARCHAR, cm.message_id::VARCHAR, cm.message, cm.is_user_message, cm.message_time,
            cm.message_author,
              COALESCE(nf.unread, FALSE) as unread
            FROM agent.chat_messages cm
            LEFT JOIN agent.notifications nf
            ON cm.message_id = nf.message_id
            WHERE cm.agent_id = ANY(%(agent_ids)s)
            ORDER BY cm.message_time ASC;
        """

        rows = self.generic_read(sql, params={"agent_ids": agent_ids})
        res: DefaultDict[str, ChatContext] = defaultdict(ChatContext)
        for row in rows:
            res[row["agent_id"]].messages.append(Message(**row))
        return res

    def get_workspace_linked_id(self, workspace_id: str) -> Optional[str]:
        sql = """
        SELECT linked_portfolio_id
        FROM workspace.workspace
        WHERE workspace_id = %(workspace_id)s
        """
        rows = self.generic_read(sql, params={"workspace_id": workspace_id})
        return rows[0]["linked_portfolio_id"] if rows else None


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
    def __init__(self, skip_commit: bool = False) -> None:
        self.db = get_psql(skip_commit=skip_commit)

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

    @contextmanager
    def cursor(self) -> Iterator[Cursor]:
        with self.db.transaction_cursor() as cursor:
            yield cursor

    async def insert_atomic(self, to_insert: List[InsertToTableArgs]) -> None:
        with self.db.transaction_cursor() as cursor:
            for arg in to_insert:
                sql, params = self.db._gen_multi_row_insert(
                    table_name=arg.table_name, values_to_insert=arg.rows, ignore_conficts=False
                )
                cursor.execute(sql, params)  # type: ignore
