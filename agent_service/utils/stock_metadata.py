from typing import Any, Dict, List, Optional

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

    rows = await get_stock_metadata_rows(gbi_ids=gbi_ids, pg=pg)
    return {row["gbi_id"]: StockMetadata(**row) for row in rows}


async def get_stock_metadata_rows(
    gbi_ids: List[int], pg: Optional[BoostedPG] = None
) -> List[Dict[str, Any]]:
    if not pg:
        # shoudl this be AsyncPostgresBase ??
        from agent_service.utils.postgres import SyncBoostedPG

        pg = SyncBoostedPG()

    sql = """
        SELECT DISTINCT ON (gbi_id)
        gbi_id,
        msc.symbol,
        msc.name AS company_name,
        msc.isin,
        msc.exchange,
        gics1.name AS sector,
        gics4.name AS subindustry,
        msc.country,
        msc.country_of_domicile,
        msc.currency,
        gics1.name AS gics1_name,
        gics2.name AS gics2_name,
        gics3.name AS gics3_name,
        gics4.name AS gics4_name,
        gics1_sector,
        gics2_industry_group,
        gics3_industry,
        gics4_sub_industry
        FROM (
        SELECT gbi_security_id AS gbi_id,
            ms.name,
            ssm.exchange,
            ms.symbol,
            ms.security_region AS country,
            ms.region AS country_of_domicile,
            ms.isin AS isin,
            ms.currency AS currency,
            ms.gics AS gics4_sub_industry,
            ms.gics / 100 AS gics3_industry,
            ms.gics / 10000 AS gics2_industry_group,
            ms.gics / 1000000 AS gics1_sector,
            ms.is_primary_trading_item AS ms_is_primary,
            ssm.is_primary_trading_item AS ssm_is_primary
        FROM master_security ms
        JOIN spiq_security_mapping ssm
            ON ssm.gbi_id = ms.gbi_security_id
        WHERE ms.gbi_security_id = ANY(%(stock_ids)s)
        AND ms.is_public
        ) msc
        LEFT JOIN gic_sector gics1 ON  gics1.id = gics1_sector
        LEFT JOIN gic_sector gics2 ON  gics2.id = gics2_industry_group
        LEFT JOIN gic_sector gics3 ON  gics3.id = gics3_industry
        LEFT JOIN gic_sector gics4 ON  gics4.id = gics4_sub_industry
        ORDER BY gbi_id, ms_is_primary DESC, ssm_is_primary DESC
    """
    rows = await pg.generic_read(
        sql,
        params={
            "stock_ids": gbi_ids,
        },
    )

    return rows
