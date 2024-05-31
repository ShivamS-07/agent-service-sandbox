from typing import Dict, List, Optional

from pydantic import BaseModel

from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.postgres import SyncBoostedPG


class StockMetadata(BaseModel):
    gbi_id: int
    symbol: Optional[str]
    isin: str
    company_name: str


# TODO cache this
async def get_stock_metadata(
    gbi_ids: List[int], pg: Optional[BoostedPG] = None
) -> Dict[int, StockMetadata]:
    pg = pg or SyncBoostedPG()
    sql = """
    SELECT gbi_security_id AS gbi_id, symbol, name AS company_name, isin
    FROM master_security
    WHERE gbi_security_id = ANY(%(gbi_ids)s)
    """

    rows = await pg.generic_read(sql, {"gbi_ids": gbi_ids})
    return {row["gbi_id"]: StockMetadata(**row) for row in rows}
