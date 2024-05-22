from typing import Dict, List

from pydantic.main import BaseModel

from agent_service.utils.boosted_pg import BoostedPG


class StockMetadata(BaseModel):
    gbi_id: int
    symbol: str
    company_name: str


# TODO cache this
async def get_stock_metadata(pg: BoostedPG, gbi_ids: List[int]) -> Dict[int, StockMetadata]:
    sql = """
    SELECT gbi_security_id AS gbi_id, symbol, name AS company_name
    FROM master_security
    WHERE gbi_security_id = ANY(%(gbi_ids)s)
    """

    rows = await pg.generic_read(sql, {"gbi_ids": gbi_ids})
    return {row["gbi_id"]: StockMetadata(**row) for row in rows}
