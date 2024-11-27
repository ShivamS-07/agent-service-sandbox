import argparse
import datetime
import json
import time
from pprint import pprint
from typing import Any, Optional

import boto3
from gbi_common_py_utils.utils.environment import PROD_TAG, get_environment_tag
from pydantic import BaseModel

from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import Postgres, get_psql


class AgentPlanRun(BaseModel):
    agent_id: str
    plan_run_id: str
    message: str

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AgentPlanRun):
            return False
        return (self.agent_id, self.plan_run_id, self.message) == (
            other.agent_id,
            other.plan_run_id,
            other.message,
        )

    def __hash__(self) -> int:
        return hash((self.agent_id, self.plan_run_id, self.message))


def update_stuck_agent_status(pr: AgentPlanRun, pg: Postgres) -> None:
    pg.generic_update(
        "AGENT.PLAN_RUNS",
        {
            "agent_id": pr.agent_id,
            "plan_run_id": pr.plan_run_id,
            "status": "RUNNING",
        },
        {
            "status": "ERROR",
        },
    )

    pg.generic_update(
        "AGENT.TASK_RUNS",
        {
            "agent_id": pr.agent_id,
            "plan_run_id": pr.plan_run_id,
            "status": "RUNNING",
        },
        {
            "status": "ERROR",
        },
    )


def get_plan_runs_stuck_greater_than(
    duration: datetime.timedelta, pg: Postgres
) -> list[AgentPlanRun]:
    sql = """
    WITH runs AS (
      SELECT DISTINCT ON (pr.agent_id) pr.agent_id, pr.plan_run_id, pr.status, pr.last_updated
      FROM agent.plan_runs pr
      JOIN agent.agents a ON a.agent_id = pr.agent_id
      AND a.automation_enabled
      ORDER BY pr.agent_id, pr.last_updated DESC
    )
    SELECT DISTINCT ON (runs.plan_run_id) runs.agent_id::TEXT, runs.plan_run_id::TEXT, al.message FROM runs
    JOIN boosted_dag.audit_log al ON runs.plan_run_id::TEXT = al.plan_run_id
    WHERE runs.status = 'RUNNING'
    AND runs.last_updated < %(limit_time)s
    """
    limit_time = get_now_utc() - duration
    rows = pg.generic_read(sql, {"limit_time": limit_time})
    return [AgentPlanRun(**row) for row in rows]


def get_plan_runs_most_recent_errored(pg: Postgres) -> list[AgentPlanRun]:
    sql = """
    WITH runs AS (
      SELECT DISTINCT ON (pr.agent_id) pr.agent_id, pr.plan_run_id, pr.status
      FROM agent.plan_runs pr
      JOIN agent.agents a ON a.agent_id = pr.agent_id
      AND a.automation_enabled
      ORDER BY pr.agent_id, pr.last_updated DESC
    )
    SELECT DISTINCT ON (runs.plan_run_id) runs.agent_id::TEXT, runs.plan_run_id::TEXT, al.message FROM runs
    JOIN boosted_dag.audit_log al ON runs.plan_run_id::TEXT = al.plan_run_id
    WHERE runs.status = 'ERROR'
    """
    rows = pg.generic_read(sql)
    return [AgentPlanRun(**row) for row in rows]


def get_plan_runs_by_id(plan_run_ids: list[str], pg: Postgres) -> list[AgentPlanRun]:
    sql = """
    SELECT DISTINCT ON (plan_run_id) agent_id::TEXT, plan_run_id::TEXT, message
    FROM boosted_dag.audit_log
    WHERE plan_run_id = ANY(%(plan_run_ids)s)
    ORDER BY plan_run_id, started_at DESC
    """
    rows = pg.generic_read(sql, {"plan_run_ids": plan_run_ids})
    return [AgentPlanRun(**row) for row in rows]


def get_most_recent_plan_runs_by_agent_id(agent_ids: list[str], pg: Postgres) -> list[AgentPlanRun]:
    sql = """
    SELECT DISTINCT ON (agent_id, plan_run_id) agent_id::TEXT, plan_run_id::TEXT, message
    FROM boosted_dag.audit_log
    WHERE agent_id = ANY(%(agent_ids)s)
    ORDER BY agent_id, plan_run_id, started_at DESC
    """
    rows = pg.generic_read(sql, {"agent_id": agent_ids})
    return [AgentPlanRun(**row) for row in rows]


def replay_plan_runs(
    runs: list[AgentPlanRun],
    queue: str,
    pg: Postgres,
    dry_run: bool = False,
    version: Optional[str] = None,
    no_gpt_cache: bool = False,
) -> None:
    sqs = boto3.resource("sqs", region_name="us-west-2")
    print(f"Using SQS queue: '{queue}'\n")
    queue = sqs.get_queue_by_name(QueueName=queue)
    for run in runs:
        print(f"Updating status for {run.plan_run_id=}")
        update_stuck_agent_status(pr=run, pg=pg)
        message = json.loads(run.message)
        if no_gpt_cache:
            message["no_gpt_cache"] = True
        if "retry_id" in message:
            del message["retry_id"]
        if version:
            message["service_versions"] = {"agent_service_version": version}
        if dry_run:
            print("Would have sent:")
            pprint(message)
        else:
            print("Sending message:")
            pprint(message)
            queue.send_message(MessageBody=json.dumps(message), DelaySeconds=0)
            time.sleep(1)
        print("\n----------\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan-run-ids", nargs="+", default=[])
    parser.add_argument(
        "-a",
        "--agent-ids",
        nargs="+",
        default=[],
        help="Will rerun the most recent runs for these agent ID's.",
    )
    parser.add_argument("-d", "--dry-run", action="store_true", default=False)
    parser.add_argument(
        "-e",
        "--all-errored",
        action="store_true",
        default=False,
        help="If set, will rerun all LIVE agents where the most recent plan run resulted in an ERROR",
    )
    parser.add_argument(
        "-s",
        "--retry-stuck-minutes",
        type=int,
        help="If set, will cancel and retry all LIVE plan runs stuck for more than the specified number of minutes",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="If set, will use the specified version when rerunning",
    )
    parser.add_argument(
        "-x",
        "--no-gpt-cache",
        action="store_true",
        default=False,
        help="If set, will bypass the GPT cache on rerun",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pg = get_psql(skip_commit=args.dry_run)
    dag_queue = (
        "insights-backend-prod-boosted-dag"
        if get_environment_tag() == PROD_TAG
        else "insights-backend-dev-boosted-dag"
    )
    runs = set()
    if args.retry_stuck_minutes:
        res = get_plan_runs_stuck_greater_than(
            duration=datetime.timedelta(minutes=args.retry_stuck_minutes), pg=pg
        )
        print(f"Found {len(res)} stuck runs to rerun: {res}")
        runs.update(res)
    if args.all_errored:
        res = get_plan_runs_most_recent_errored(pg=pg)
        print(f"Found {len(res)} runs that errored to rerun: {res}")
        runs.update(res)
    if args.plan_run_ids:
        print(f"Fetching data for {len(args.plan_run_ids)} manually specified plan runs")
        runs.update(get_plan_runs_by_id(plan_run_ids=args.plan_run_ids, pg=pg))
    if args.agent_ids:
        res = get_most_recent_plan_runs_by_agent_id(agent_ids=args.agent_ids, pg=pg)
        print(f"Found {len(res)} most recent runs for agents to rerun: {res}")
        runs.update(res)

    if not runs:
        print("No runs found! Exiting.")
        return
    print("\n#############################")
    print("Starting Reruns!!")
    print("#############################\n")
    replay_plan_runs(
        list(runs),
        queue=dag_queue,
        pg=pg,
        dry_run=args.dry_run,
        version=args.version,
        no_gpt_cache=args.no_gpt_cache,
    )


if __name__ == "__main__":
    init_stdout_logging()
    main()
