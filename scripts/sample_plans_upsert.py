import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from gbi_common_py_utils.utils.environment import DEV_TAG, PROD_TAG
from gbi_common_py_utils.utils.postgres import PostgresBase

BACKUP_DIR = "backups"
DATE_FORMAT = "%Y%m%d%H%M%S"
TABLE_NAME = "agent.sample_plans"


def backup_plans(
    plans: List[Dict], environment: str, backup_file_path: Optional[str] = None
) -> str:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime(DATE_FORMAT)
    if backup_file_path is None:
        backup_file = os.path.join(
            BACKUP_DIR, f"{environment}_{TABLE_NAME}_backup_{timestamp}.json"
        )
    else:
        backup_file = backup_file_path
    with open(backup_file, "w") as file:
        json.dump(plans, file, default=str)
    return backup_file


def get_plans_from_environment(environment: str, skip_commit: bool) -> List[Dict]:
    print(f"Fetching plans from {environment} environment")
    sql = f"""
    SELECT sample_plan_id, input, plan, created_at
    FROM {TABLE_NAME}
    """
    db = PostgresBase(environment=environment, skip_commit=skip_commit)
    results = db.generic_read(sql)
    print(f"Found {len(results)} plans in {environment}")
    return results


def sync_plans_into_env(plans: List[Dict], environment: str, skip_commit: bool) -> None:
    print(f"Syncing plans into {environment} environment")
    db = PostgresBase(environment=environment, skip_commit=skip_commit)

    # do any transforms if needed
    rows = [
        {
            "sample_plan_id": r["sample_plan_id"],
            "input": r["input"],
            "plan": r["plan"],
            "created_at": r["created_at"],
        }
        for r in plans
    ]

    try:
        with db.transaction_cursor() as cursor:
            # Delete existing records
            delete_sql = f"DELETE FROM {TABLE_NAME}"
            cursor.execute(delete_sql)

            # Insert new records
            insert_sql_and_params = db._gen_multi_row_insert(
                table_name=TABLE_NAME, values_to_insert=rows, ignore_conficts=False
            )
            cursor.execute(*insert_sql_and_params)
    except Exception as e:
        print(f"Failed to sync rows: {repr(e)}")


def restore_plans_from_backup(backup_file: str, environment: str, skip_commit: bool) -> None:
    print(f"Restoring plans from backup file: {backup_file}")
    with open(backup_file, "r") as file:
        plans = json.load(file)
    sync_plans_into_env(plans, environment, skip_commit)
    print(f"Restored plans from {backup_file} into {environment} environment")


def main(args: argparse.Namespace) -> None:
    if args.action == "backup_and_upsert":
        # Backup production plans
        prod_plans = get_plans_from_environment(PROD_TAG, args.skip_commit)
        backup_file = backup_plans(prod_plans, PROD_TAG, args.backup_file)
        print(f"Backed up production plans to {backup_file}")

        # Fetch development plans
        dev_plans = get_plans_from_environment(DEV_TAG, args.skip_commit)

        # Sync development plans into production
        sync_plans_into_env(dev_plans, PROD_TAG, args.skip_commit)
        print("Synced development plans into production")

    elif args.action == "restore_from_backup":
        if not args.backup_file:
            raise ValueError("Backup file path must be provided for restore action")
        restore_plans_from_backup(args.backup_file, PROD_TAG, args.skip_commit)

    else:
        raise ValueError(f"Unknown action: {args.action}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upsert/Overwrite plans script with backup and restore options"
    )
    parser.add_argument(
        "--action",
        choices=["backup_and_upsert", "restore_from_backup"],
        help="Action to perform",
        default="backup_and_upsert",
    )
    parser.add_argument("--backup_file", type=str, help="Path to backup file for restore action")
    parser.add_argument(
        "--skip_commit", action="store_true", help="Skip commit for database operations"
    )

    args = parser.parse_args()

    main(args)
