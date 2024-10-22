import argparse
import asyncio
import datetime
import json
import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple

from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.event_logging import log_event
from agent_service.utils.sec.constants import FILE_10K, FILE_10Q, FILINGS
from agent_service.utils.sec.sec_api import SecFiling

LOGGER = logging.getLogger(__name__)


async def main(
    gbi_ids: List[int] = [],
    num_workers: int = 1,
    worker: int = 0,
    start_date: datetime.date = datetime.date(1900, 1, 1),
    end_date: datetime.date = datetime.date.today() + datetime.timedelta(days=30),
) -> None:
    ch = Clickhouse()
    sql = """
    select gbi_id, cik, filing from sec.sec_filings
    """
    where_clause = ""
    if gbi_ids:
        where_clause = "where gbi_id in %(gbi_ids)s"
    rows = await ch.generic_read(
        sql + where_clause, params=({"gbi_ids": gbi_ids} if gbi_ids else None)
    )
    cik_gbi_map = {}
    filing_set = set()
    for row in rows:
        if row["gbi_id"] % num_workers != worker:
            continue
        cik_gbi_map[row["cik"]] = row["gbi_id"]
        filing_set.add(row["filing"])

    ciks = list(cik_gbi_map.keys())
    for cik in ciks:
        existing_filings = 0
        total_existing_filings = 0
        start = time.time()
        total_inserted = 0
        gbi_id = cik_gbi_map[cik]
        query_start_date = start_date
        query_end_date = end_date
        try:

            LOGGER.info(f"Working on gbi_id={gbi_id}")
            queries = SecFiling._build_queries_for_filings(
                cik=cik,
                form_types=[FILE_10K, FILE_10Q],
                start_date=query_start_date,
                end_date=query_end_date,
            )
            filings = []
            for query in queries:
                document_resp: Optional[Dict] = SecFiling.query_api.get_filings(query=query)
                if (
                    (not document_resp)
                    or (FILINGS not in document_resp)
                    or (not document_resp[FILINGS])
                ):
                    continue  # move onto next form type

                filings_ = document_resp[FILINGS]
                LOGGER.info(f"Got {len(filings_)} filings from the SEC API")
                to_insert_map: List[Tuple[str, int]] = []
                for filing in filings_:
                    filing_str = json.dumps(filing)
                    if filing_str not in filing_set:
                        filings.append(filing)
                        to_insert_map.append((filing_str, cik_gbi_map[filing["cik"]]))
                existing_filings = len(filings_) - len(to_insert_map)
                total_existing_filings += existing_filings
                LOGGER.info(
                    f"Inserting {len(to_insert_map)} filings into clickhouse. Already had {existing_filings} filings."
                )

                if to_insert_map:
                    start_time = time.time()
                    _, rows_to_insert = await SecFiling.get_concat_10k_10q_sections_from_api(
                        filing_gbi_pairs=to_insert_map, insert_to_db=False
                    )
                    await ch.multi_row_insert(table_name="sec.sec_filings", rows=rows_to_insert)
                    LOGGER.info(
                        f"Inserted {len(to_insert_map)} filings into Clickhouse in {time.time() - start_time} seconds."
                    )
                    total_inserted += len(to_insert_map)
                length_of_this = len(document_resp[FILINGS])
                if length_of_this < 50:
                    ingest_time_seconds = time.time() - start
                    LOGGER.info(
                        f"Done with gbi_id={gbi_id}."
                        f"Inserted {total_inserted} documents in {ingest_time_seconds} seconds."
                    )
                    log_event(
                        event_name="sec-ingest-event",
                        event_data={
                            "gbi_id": gbi_id,
                            "duration_seconds": ingest_time_seconds,
                            "num_inserted": total_inserted,
                            "existing_filings": total_existing_filings,
                            "query_start_date": query_start_date.isoformat(),
                            "query_end_date": query_end_date.isoformat(),
                        },
                    )
                    break
        except Exception:
            error_msg = traceback.format_exc()
            ingest_time_seconds = time.time() - start
            LOGGER.info(f"Error processing gbi_id={gbi_id}")
            LOGGER.info(error_msg)
            log_event(
                event_name="sec-ingest-event",
                event_data={
                    "gbi_id": gbi_id,
                    "duration_seconds": ingest_time_seconds,
                    "num_inserted": total_inserted,
                    "existing_filings": existing_filings,
                    "query_start_date": query_start_date.isoformat(),
                    "query_end_date": query_end_date.isoformat(),
                    "error_msg": error_msg,
                },
            )

            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest SEC filings data.")
    parser.add_argument(
        "--gbi-ids", nargs="+", type=int, default=[], help="List of gbi_ids to process"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of workers to run the process"
    )
    parser.add_argument(
        "--worker", type=int, default=0, help="ID of the worker running the process"
    )
    parser.add_argument(
        "--start-date",
        type=lambda x: datetime.date.fromisoformat(x),
        default=datetime.date(1900, 1, 1),
    )
    parser.add_argument(
        "--end-date",
        type=lambda x: datetime.date.fromisoformat(x),
        default=datetime.date.today() + datetime.timedelta(days=30),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
        force=True,
    )
    args = parse_args()
    asyncio.run(
        main(
            gbi_ids=args.gbi_ids,
            num_workers=args.num_workers,
            worker=args.worker,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    )
