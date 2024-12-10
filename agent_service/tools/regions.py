from typing import List

import country_converter as coco
from gbi_common_py_utils.utils.util import memoize_one

from agent_service.io_types.stock import StockID
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import get_async_db
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.tool_diff import (
    add_task_id_to_stocks_history,
    get_prev_run_info,
)

# seems unlikely stocks will ever change region, but I guess it is possible
REGION_ADD_STOCK_DIFF = "{company} was added to the {region} region"
REGION_REMOVE_STOCK_DIFF = "{company} was removed from the {region} region"


class FilterStockRegionInput(ToolArgs):
    stock_ids: List[StockID]
    region_name: str


class FilterStockContryOfDomicileInput(ToolArgs):
    stock_ids: List[StockID]
    country_name: str


@memoize_one
def country_converter() -> coco.CountryConverter:
    converter = coco.CountryConverter()
    return converter


def get_country_iso3s(search: str) -> List[str]:
    region_or_country = search.upper()

    if region_or_country in REGION_COUNTRY_MAP:
        countries = REGION_COUNTRY_MAP[region_or_country]
        return countries

    # Sometimes this fails because of the '_' in the region_or_country
    new_region_or_country = region_or_country.replace(" ", "_")

    # now check to see if we have any more regions
    if new_region_or_country in REGION_COUNTRY_MAP:
        countries = REGION_COUNTRY_MAP[new_region_or_country]
        return countries

    countries = country_converter().convert(names=search, to="ISO3", not_found="UNKNOWN_REGION")
    if isinstance(countries, list):
        return countries
    else:
        return [countries]


async def get_country_name(search: str) -> str:
    # create get country to iso dict
    iso_to_country_name = {
        get_country_iso3s(country)[0]: country for country in SUPPORTED_COUNTRIES
    }
    # Create a country to region map
    if search.upper() in iso_to_country_name:
        return iso_to_country_name.get(search.upper(), "UNKNOWN_COUNTRY")
    incoming_country_name = get_country_iso3s(search)[0]
    return iso_to_country_name.get(incoming_country_name, "UNKNOWN_COUNTRY")


# This REGION -> COUNTRY map contains all the countries in the
# Vanguard International Equity Index Funds - Vanguard Total World Stock ETF
# filtered to regions using https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
REGION_COUNTRY_MAP = {
    "NORTHERN_AFRICA": ["DZA", "EGY", "LBY", "MAR", "SDN", "TUN"],
    "SUB_SAHARAN_AFRICA": [
        "AGO",
        "BEN",
        "BWA",
        "BFA",
        "BDI",
        "CPV",
        "CMR",
        "CAF",
        "TCD",
        "COM",
        "COG",
        "COD",
        "CIV",
        "DJI",
        "GNQ",
        "ERI",
        "SWZ",
        "ETH",
        "GAB",
        "GMB",
        "GHA",
        "GIN",
        "GNB",
        "KEN",
        "LSO",
        "LBR",
        "MDG",
        "MWI",
        "MLI",
        "MRT",
        "MUS",
        "MOZ",
        "NAM",
        "NER",
        "NGA",
        "RWA",
        "STP",
        "SEN",
        "SYC",
        "SLE",
        "SOM",
        "ZAF",
        "SSD",
        "TZA",
        "TGO",
        "UGA",
        "ZMB",
        "ZWE",
    ],
    "CARIBBEAN": ["BHS", "BRB", "CUB", "DOM", "GRD", "HTI", "JAM", "KNA", "LCA", "VCT", "TTO"],
    "CENTRAL_AMERICA": ["BLZ", "CRI", "SLV", "GTM", "HND", "NIC", "PAN", "MEX"],
    "NORTH_AMERICA": ["CAN", "USA"],
    "SOUTH_AMERICA": [
        "ARG",
        "BOL",
        "BRA",
        "CHL",
        "COL",
        "ECU",
        "GUY",
        "PRY",
        "PER",
        "SUR",
        "URY",
        "VEN",
    ],
    "EASTERN_ASIA": ["CHN", "HKG", "JPN", "KOR", "MNG", "PRK", "MAC", "TWN"],
    "SOUTH_EASTERN_ASIA": [
        "BRN",
        "MMR",
        "KHM",
        "IDN",
        "LAO",
        "MYS",
        "PHL",
        "SGP",
        "THA",
        "TLS",
        "VNM",
    ],
    "SOUTHERN_ASIA": ["AFG", "BGD", "BTN", "IND", "MDV", "NPL", "LKA", "PAK"],
    "WESTERN_ASIA": [
        "ARM",
        "AZE",
        "BHR",
        "CYP",
        "GEO",
        "IRQ",
        "ISR",
        "JOR",
        "KWT",
        "LBN",
        "OMN",
        "QAT",
        "SAU",
        "SYR",
        "TUR",
        "ARE",
        "YEM",
        "PSE",
    ],
    "CENTRAL_ASIA": ["KAZ", "KGZ", "TJK", "TKM", "UZB"],
    "EASTERN_EUROPE": ["BLR", "BGR", "HUN", "MDA", "POL", "ROU", "RUS", "SVK", "UKR"],
    "NORTHERN_EUROPE": [
        "DNK",
        "EST",
        "FIN",
        "FRO",
        "ISL",
        "IRL",
        "LVA",
        "LTU",
        "NLD",
        "SJM",
        "SWE",
        "GBR",
    ],
    "SOUTHERN_EUROPE": [
        "ALB",
        "AND",
        "ARM",
        "AUT",
        "BGR",
        "CYP",
        "FRA",
        "GIB",
        "GRE",
        "ITA",
        "MCO",
        "MLT",
        "PRT",
        "SMR",
        "ESP",
        "VAT",
    ],
    "WESTERN_EUROPE": ["AND", "AUT", "BEL", "FRA", "DEU", "LUX", "MCO", "NLD", "CHE"],
    "OCEANIA": [
        "AUS",
        "FJI",
        "KIR",
        "MSR",
        "NCL",
        "NZL",
        "PLW",
        "PNG",
        "SAM",
        "SLB",
        "TON",
        "TUV",
        "VUT",
    ],
    "AUSTRALIA_AND_NEW_ZEALAND": ["AUS", "NZL"],
    "EU_EUROPEAN_UNION": [
        "AUT",
        "BEL",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "IRL",
        "ITA",
        "LVA",
        "LTU",
        "LUX",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "ESP",
        "SWE",
    ],
    "NATO": [
        "ALB",
        "BEL",
        "BGR",
        "CAN",
        "HRV",
        "CZE",
        "DNK",
        "EST",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "ISL",
        "ITA",
        "LVA",
        "LTU",
        "LUX",
        "MNE",
        "NLD",
        "MKD",
        "NOR",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "ESP",
        "TUR",
        "GBR",
        "USA",
    ],
}
REGION_COUNTRY_MAP["AFRICA"] = list(
    set(REGION_COUNTRY_MAP["NORTHERN_AFRICA"] + REGION_COUNTRY_MAP["SUB_SAHARAN_AFRICA"])
)
REGION_COUNTRY_MAP["LATIN_AMERICA"] = list(
    set(
        REGION_COUNTRY_MAP["CARIBBEAN"]
        + REGION_COUNTRY_MAP["SOUTH_AMERICA"]
        + REGION_COUNTRY_MAP["CENTRAL_AMERICA"]
    )
)
REGION_COUNTRY_MAP["AMERICAS"] = list(
    set(REGION_COUNTRY_MAP["NORTH_AMERICA"] + REGION_COUNTRY_MAP["LATIN_AMERICA"])
)
REGION_COUNTRY_MAP["MIDDLE_EAST"] = REGION_COUNTRY_MAP["WESTERN_ASIA"]
REGION_COUNTRY_MAP["ASIA"] = list(
    set(
        REGION_COUNTRY_MAP["WESTERN_ASIA"]
        + REGION_COUNTRY_MAP["CENTRAL_ASIA"]
        + REGION_COUNTRY_MAP["EASTERN_ASIA"]
        + REGION_COUNTRY_MAP["SOUTH_EASTERN_ASIA"]
        + REGION_COUNTRY_MAP["SOUTHERN_ASIA"]
    )
)
REGION_COUNTRY_MAP["EUROPE"] = list(
    set(
        REGION_COUNTRY_MAP["WESTERN_EUROPE"]
        + REGION_COUNTRY_MAP["NORTHERN_EUROPE"]
        + REGION_COUNTRY_MAP["EASTERN_EUROPE"]
        + REGION_COUNTRY_MAP["SOUTHERN_EUROPE"]
    )
)
REGION_COUNTRY_MAP["APAC_ASIA_PACIFIC"] = list(
    set(REGION_COUNTRY_MAP["EASTERN_ASIA"] + REGION_COUNTRY_MAP["OCEANIA"])
)

# A list of all the countries that are supported in the master_security tables
SUPPORTED_COUNTRIES = [
    "Albania",
    "Anguilla",
    "Argentina",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bahrain",
    "Bangladesh",
    "Barbados",
    "Belgium",
    "Belize",
    "Bermuda",
    "Bolivia",
    "Bosnia-Herzegovina",
    "Botswana",
    "Brazil",
    "British Virgin Islands",
    "Bulgaria",
    "Burkina Faso",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Cayman Islands",
    "Channel Islands",
    "Chile",
    "China",
    "Colombia",
    "Costa Rica",
    "Croatia",
    "Curaçao",
    "Cyprus",
    "Czech Republic",
    "Democratic Republic",
    "Denmark",
    "Dominican Republic",
    "Egypt",
    "El Salvador",
    "Estonia",
    "Falkland Islands",
    "Finland",
    "France",
    "French Guiana",
    "Gabon",
    "Georgia",
    "Germany",
    "Ghana",
    "Gibraltar",
    "Greece",
    "Greenland",
    "Guernsey",
    "Honduras",
    "Hong Kong",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Ireland",
    "Isle of Man",
    "Israel",
    "Italy",
    "Ivory Coast",
    "Jamaica",
    "Japan",
    "Jersey",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kuwait",
    "Kyrgyzstan",
    "Latvia",
    "Lebanon",
    "Liberia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Macau",
    "Macedonia",
    "Malaysia",
    "Malta",
    "Marshall Islands",
    "Martinique",
    "Mauritius",
    "Mexico",
    "Monaco",
    "Mongolia",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Myanmar",
    "Namibia",
    "Netherlands",
    "Netherlands Antilles",
    "New Zealand",
    "Nigeria",
    "Norway",
    "Oman",
    "Pakistan",
    "Palestinian Authority",
    "Panama",
    "Papua New Guinea",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Reunion",
    "Romania",
    "Russia",
    "Saint Kitts & Nevis",
    "Saint Vincent & Grenadines",
    "Samoa",
    "Saudi Arabia",
    "Senegal",
    "Serbia",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "South Africa",
    "South Korea",
    "Spain",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Sweden",
    "Switzerland",
    "Taiwan",
    "Tanzania",
    "Thailand",
    "Trinidad & Tobago",
    "Tunisia",
    "Turkey",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Uruguay",
    "Venezuela",
    "Vietnam",
    "Zambia",
    "Zimbabwe",
]

COUNTRY_MAIN_PROMPT = Prompt(
    name="LLM_COUNTRY_MAIN_PROMPT",
    template=(
        "Match the string to a country in the list of countries provided."
        "The string is as follows:\n"
        "{raw_country_string}\n"
        "Return only the country string as found in the list"
        "If and ONLY IF you can not find the country in list return your best guess"
    ),
)
COUNTRY_SYS_PROMPT = Prompt(
    name="LLM_COUNTRY_SYS_PROMPT",
    template=(
        "You are an assistant designed to process a string that refers to some kind of country"
        "and convert it into a known country for example if the passed string is Cad you will return Canada"
        "Return only the country name as a string with the first letter of every word MUST capitalized"
        "Do not add anything to the answer"
        "Only return countries found in the following country list {country_list}"
    ),
)


@tool(
    description=(
        "This function takes a list of stock ID's and either an ISO 3166-1 alpha-3 country code string"
        " (e.g., 'USA', 'CHE', 'JPN', 'CAN', 'GBR'), or a country name in english, or a specific region name."
        " It filters the list of stocks by the given country or region. Supported regions are: "
        + ", ".join(REGION_COUNTRY_MAP.keys())
        + "."
        " For countries, use the standard 3-letter ISO country code."
        " The function returns the filtered list of stock IDs."
    ),
    category=ToolCategory.STOCK_FILTERS,
    tool_registry=default_tool_registry(),
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

    logger = get_prefect_logger(__name__)

    # GPT is passing in some non-iso3 strings sometimes like FRANCE
    # this lib is really good at converting arbitrary country mentions to ISO codes
    countries = get_country_iso3s(args.region_name)

    if countries and countries[0].upper() != args.region_name.upper():
        await tool_log(
            log=f"Interpreting '{args.region_name.upper()}' as {countries}", context=context
        )

    sql = """
    SELECT gbi_security_id
    FROM master_security
    WHERE security_region = ANY(%(countries)s)
    AND gbi_security_id = ANY(%(stocks)s)
    """
    async_db = get_async_db()
    rows = await async_db.generic_read(
        sql, {"stocks": [stock.gbi_id for stock in args.stock_ids], "countries": countries}
    )
    stocks_to_include = {row["gbi_security_id"] for row in rows}
    num_filtered_stocks = len(stocks_to_include)
    if num_filtered_stocks != 0:
        await tool_log(
            log=(
                f"Filtered {len(args.stock_ids)} stocks by region down to "
                f"{num_filtered_stocks} stocks for {args.region_name}"
            ),
            context=context,
        )
    stock_list = [stock for stock in args.stock_ids if stock.gbi_id in stocks_to_include]
    if not stock_list:
        raise EmptyOutputError(message="Stock filter resulted in an empty list of stocks")

    try:  # since everything associated with diffing is optional, put in try/except
        # we need to add the task id to all runs, including the first one, so we can track changes
        if context.task_id:
            stock_list = add_task_id_to_stocks_history(stock_list, context.task_id)
            if context.diff_info is not None:
                # 2nd arg is the name of the function we are in
                prev_run_info = await get_prev_run_info(context, "filter_stocks_by_region")
                if prev_run_info is not None:
                    prev_input = FilterStockRegionInput.model_validate_json(
                        prev_run_info.inputs_str
                    )
                    prev_output: List[StockID] = prev_run_info.output  # type:ignore
                    # corner case here where S&P 500 change causes output to change, but not going to
                    # bother with it on first pass
                    if args.stock_ids and prev_input.stock_ids:
                        # we only care about stocks that were inputs for both
                        shared_inputs = set(prev_input.stock_ids) & set(args.stock_ids)
                    else:
                        shared_inputs = set()
                    curr_stock_set = set(stock_list)
                    prev_stock_set = set(prev_output)
                    added_stocks = (curr_stock_set - prev_stock_set) & shared_inputs
                    removed_stocks = (prev_stock_set - curr_stock_set) & shared_inputs
                    context.diff_info[context.task_id] = {
                        "added": {
                            added_stock: REGION_ADD_STOCK_DIFF.format(
                                company=added_stock.company_name, region=args.region_name
                            )
                            for added_stock in added_stocks
                        },
                        "removed": {
                            removed_stock: REGION_REMOVE_STOCK_DIFF.format(
                                company=removed_stock.company_name, region=args.region_name
                            )
                            for removed_stock in removed_stocks
                        },
                    }

    except Exception as e:
        logger.warning(f"Error creating diff info from previous run: {e}")

    return stock_list


@tool(
    description=(
        "This function takes a list of stock ID's and the full country name"
        " (e.g., 'United States', 'Canada', China) and It filters the list of stocks by the given country"
        " The function returns the filtered list of stock IDs."
    ),
    category=ToolCategory.STOCK_FILTERS,
    tool_registry=default_tool_registry(),
)
async def filter_stocks_by_country_of_domicile(
    args: FilterStockContryOfDomicileInput, context: PlanRunContext
) -> List[StockID]:
    """
    Returns: List[StockId]
    """

    logger = get_prefect_logger(__name__)

    country = await get_country_name(args.country_name)
    await tool_log(
        log=(f"{args.country_name} is interpreted as {country}"),
        context=context,
    )

    sql = """
    SELECT gbi_security_id
    FROM master_security
    WHERE region = %(country)s
    AND gbi_security_id = ANY(%(stocks)s)
    """
    db = get_psql()
    rows = db.generic_read(
        sql, {"stocks": [stock.gbi_id for stock in args.stock_ids], "country": country}
    )
    stocks_to_include = {row["gbi_security_id"] for row in rows}
    num_filtered_stocks = len(stocks_to_include)
    if num_filtered_stocks != 0:
        await tool_log(
            log=(
                f"Filtered {len(args.stock_ids)} stocks by region down to "
                f"{num_filtered_stocks} stocks for {args.country_name}"
            ),
            context=context,
        )
    stock_list = [stock for stock in args.stock_ids if stock.gbi_id in stocks_to_include]
    if not stock_list:
        raise EmptyOutputError(message="Stock filter resulted in an empty list of stocks")

    try:  # since everything associated with diffing is optional, put in try/except
        # we need to add the task id to all runs, including the first one, so we can track changes
        if context.task_id:
            stock_list = add_task_id_to_stocks_history(stock_list, context.task_id)
            if context.diff_info is not None:
                # 2nd arg is the name of the function we are in
                prev_run_info = await get_prev_run_info(
                    context, "filter_stocks_by_country_of_domicile"
                )
                if prev_run_info is not None:
                    prev_input = FilterStockRegionInput.model_validate_json(
                        prev_run_info.inputs_str
                    )
                    prev_output: List[StockID] = prev_run_info.output  # type:ignore
                    # corner case here where S&P 500 change causes output to change, but not going to
                    # bother with it on first pass
                    if args.stock_ids and prev_input.stock_ids:
                        # we only care about stocks that were inputs for both
                        shared_inputs = set(prev_input.stock_ids) & set(args.stock_ids)
                    else:
                        shared_inputs = set()
                    curr_stock_set = set(stock_list)
                    prev_stock_set = set(prev_output)
                    added_stocks = (curr_stock_set - prev_stock_set) & shared_inputs
                    removed_stocks = (prev_stock_set - curr_stock_set) & shared_inputs
                    context.diff_info[context.task_id] = {
                        "added": {
                            added_stock: REGION_ADD_STOCK_DIFF.format(
                                company=added_stock.company_name, region=args.country_name
                            )
                            for added_stock in added_stocks
                        },
                        "removed": {
                            removed_stock: REGION_REMOVE_STOCK_DIFF.format(
                                company=removed_stock.company_name, region=args.country_name
                            )
                            for removed_stock in removed_stocks
                        },
                    }

    except Exception as e:
        logger.warning(f"Error creating diff info from previous run: {e}")

    return stock_list
