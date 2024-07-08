from typing import List

from agent_service.io_types.stock import StockID
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql


class FilterStockRegionInput(ToolArgs):
    stock_ids: List[StockID]
    region_name: str


# This REGION -> COUNTRY map contains all the countries in the
# Vanguard International Equity Index Funds - Vanguard Total World Stock ETF
# filtered to regions using https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
REGION_COUNTRY_MAP = {
    "AFRICA": ["EGY", "ZAF"],
    "NORTHERN_AFRICA": ["EGY"],
    "SUB_SAHARAN_AFRICA": ["ZAF"],
    "AMERICAS": ["MEX", "CHL", "BRA", "COL", "USA", "CAN"],
    "LATIN_AMERICA": ["BRA", "CHL", "COL", "MEX"],
    "NORTH_AMERICA": ["CAN", "USA"],
    "SOUTH_AMERICA": ["BRA", "CHL", "COL"],
    "ASIA": [
        "KOR",
        "MYS",
        "JPN",
        "SGP",
        "KWT",
        "THA",
        "SAU",
        "ARE",
        "QAT",
        "HKG",
        "PHL",
        "CHN",
        "IND",
        "IDN",
        "TUR",
        "ISR",
        "PAK",
        "VNM",
    ],
    "EASTERN_ASIA": ["CHN", "HKG", "JPN", "KOR"],
    "SOUTH_EASTERN_ASIA": ["IDN", "MYS", "PHL", "SGP", "THA", "VNM"],
    "SOUTHERN_ASIA": ["IND", "PAK"],
    "WESTERN_ASIA": ["ISR", "KWT", "QAT", "SAU", "TUR", "ARE"],
    "MIDDLE_EAST": ["ISR", "KWT", "QAT", "SAU", "TUR", "ARE"],
    "EUROPE": [
        "AUT",
        "GRC",
        "POL",
        "SWE",
        "DEU",
        "IRL",
        "BEL",
        "FIN",
        "PRT",
        "DNK",
        "CHE",
        "FRA",
        "CZE",
        "ITA",
        "NOR",
        "ISL",
        "NLD",
        "GBR",
        "RUS",
        "ESP",
        "LUX",
        "HUN",
    ],
    "EASTERN_EUROPE": ["CZE", "HUN", "POL", "RUS"],
    "NORTHERN_EUROPE": ["DNK", "FIN", "ISL", "IRL", "NOR", "SWE", "GBR"],
    "SOUTHERN_EUROPE": ["GRC", "ITA", "PRT", "ESP"],
    "WESTERN_EUROPE": ["AUT", "BEL", "FRA", "DEU", "LUX", "NLD", "CHE"],
    "OCEANIA": ["AUS", "NZL"],
    "AUSTRALIA_AND_NEW_ZEALAND": ["AUS", "NZL"],
    "APAC_ASIA_PACIFIC": [
        "KOR",
        "MYS",
        "JPN",
        "SGP",
        "KWT",
        "THA",
        "SAU",
        "ARE",
        "QAT",
        "HKG",
        "PHL",
        "CHN",
        "IND",
        "IDN",
        "TUR",
        "ISR",
        "PAK",
        "VNM",
        "AUS",
        "NZL",
    ],
    "EU_EUROPEAN_UNION": [
        "AUT",
        "BEL",
        "CZE",
        "DNK",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "IRL",
        "ITA",
        "LUX",
        "NLD",
        "POL",
        "PRT",
        "ESP",
        "SWE",
    ],
    "NATO": [
        "CAN",
        "USA",
        "BEL",
        "DNK",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "ISL",
        "ITA",
        "LUX",
        "NLD",
        "NOR",
        "POL",
        "PRT",
        "ESP",
        "TUR",
        "GBR",
        "CZE",
    ],
}


@tool(
    description=(
        "This function takes a list of stock ID's and either an ISO3 country code string"
        " (e.g., 'USA', 'CAN') or a specific region name. It filters the list of stocks by the given"
        " country or region. Supported regions are: " + ", ".join(REGION_COUNTRY_MAP.keys()) + "."
        " For countries, use the standard 3-letter ISO country code."
        " The function returns the filtered list of stock IDs."
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def filter_stocks_by_region(
    args: FilterStockRegionInput, context: PlanRunContext
) -> List[StockID]:
    """
    This tool should be able to find generic regions like EUROPE
    and filter down to a list of countries.
    It should also recognize ISO3 country codes if provided

    Returns: List[StockId]
    """
    region_or_country = args.region_name.upper()

    if region_or_country in REGION_COUNTRY_MAP:
        countries = REGION_COUNTRY_MAP[region_or_country]
    else:
        countries = [region_or_country]

    sql = """
    SELECT gbi_security_id
    FROM master_security
    WHERE security_region = ANY(%(countries)s)
    AND gbi_security_id = ANY(%(stocks)s)
    """
    db = get_psql()
    rows = db.generic_read(
        sql, {"stocks": [stock.gbi_id for stock in args.stock_ids], "countries": countries}
    )
    stocks_to_include = {row["gbi_security_id"] for row in rows}
    num_filtered_stocks = len(stocks_to_include)
    if num_filtered_stocks == 0:
        await tool_log(log=f"No stocks filtered for {args.region_name}", context=context)
    else:
        await tool_log(
            log=f"Filtered {num_filtered_stocks} stocks for {args.region_name}", context=context
        )
    return [stock for stock in args.stock_ids if stock.gbi_id in stocks_to_include]
