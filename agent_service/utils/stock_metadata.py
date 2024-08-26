from typing import Dict, List, Optional

from pydantic import BaseModel

from agent_service.utils.boosted_pg import BoostedPG


# THIS SHOULD NOT BE USED INTERNALLY, ONLY FOR SENDING TO FRONTEND
class StockMetadata(BaseModel):
    gbi_id: int
    symbol: Optional[str]
    company_name: str
    isin: str = ""
    sector: Optional[str] = None
    subindustry: Optional[str] = None
    exchange: Optional[str] = None


# TODO cache this
async def get_stock_metadata(
    gbi_ids: List[int], pg: Optional[BoostedPG] = None
) -> Dict[int, StockMetadata]:
    if not pg:
        from agent_service.utils.postgres import SyncBoostedPG

        pg = SyncBoostedPG()
    sql = """
    SELECT gbi_id,
        msc.symbol,
        msc.name AS company_name,
        isin,
        exchange,
        gics1.name AS sector,
        gics4.name AS subindustry
        FROM (
        SELECT gbi_security_id AS gbi_id,
            ssm.exchange,
            ms.isin AS isin,
            ms.symbol,
            ms.name,
            ms.currency AS currency,
            ms.gics AS gics4_sub_industry,
            ms.gics / 100 AS gics3_industry,
            ms.gics / 10000 AS gics2_industry_group,
            ms.gics / 1000000 AS gics1_sector
    FROM master_security ms
    JOIN spiq_security_mapping ssm
       ON ssm.gbi_id = ms.gbi_security_id
    WHERE ms.gbi_security_id = ANY(%(gbi_ids)s)
    ) msc
    LEFT JOIN gic_sector gics1 ON  gics1.id = gics1_sector
    LEFT JOIN gic_sector gics2 ON  gics2.id = gics2_industry_group
    LEFT JOIN gic_sector gics3 ON  gics3.id = gics3_industry
    LEFT JOIN gic_sector gics4 ON  gics4.id = gics4_sub_industry;
    """

    rows = await pg.generic_read(sql, {"gbi_ids": gbi_ids})
    return {row["gbi_id"]: StockMetadata(**row) for row in rows}
