from typing import List

import country_converter as coco
from gbi_common_py_utils.utils.util import memoize_one

from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_types.stock import StockID
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
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


async def get_country_name(search: str, retries: int = 3) -> str:
    llm = GPT(model=DEFAULT_CHEAP_MODEL)
    result = search
    for _ in range(retries):
        result = await llm.do_chat_w_sys_prompt(
            main_prompt=COUNTRY_MAIN_PROMPT.format(raw_country_string=search),
            sys_prompt=COUNTRY_SYS_PROMPT.format(
                country_list="" + ", ".join(SUPPORTED_COUNTRIES) + "."
            ),
        )
        # check if a correct country was returned
        if result in SUPPORTED_COUNTRIES:
            break
    return result


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
REGION_COUNTRY_MAP["AFRICA"] = list(
    set(REGION_COUNTRY_MAP["NORTHERN_AFRICA"] + REGION_COUNTRY_MAP["SUB_SAHARAN_AFRICA"])
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
        "This function takes a list of stock ID's and either an ISO3 country code string"
        " (e.g., 'USA', 'CAN', 'JPN'), a country's name in english,"
        " or a specific region name which represents a group of countries."
        " It filters the list of stocks by the given"
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
    db = get_psql()
    rows = db.generic_read(
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
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
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
