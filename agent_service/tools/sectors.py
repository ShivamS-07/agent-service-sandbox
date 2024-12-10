import inspect
import json
from threading import Lock
from typing import Dict, List, Optional

from cachetools import TTLCache, cached

from agent_service.external import sec_meta_svc_client
from agent_service.GPT.constants import GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import Text
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.pagerduty import pager_wrapper
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.stock_metadata import get_stock_metadata_rows
from agent_service.utils.string_utils import repair_json_if_needed
from agent_service.utils.tool_diff import (
    add_task_id_to_stocks_history,
    get_prev_run_info,
)

ONE_HOUR = 60 * 60

SECTOR_ADD_STOCK_DIFF = "{company} was added to the {sector} sector"
SECTOR_REMOVE_STOCK_DIFF = "{company} was removed from the {sector} sector"


@io_type
class SectorID(ComplexIOBase):
    sec_id: int
    sec_name: str

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        t: Text = Text(val=f"Sector: {self.sec_name}")
        return await t.to_rich_output(pg=pg, title=title)


class GetStockSectorInput(ToolArgs):
    stock_id: StockID


@tool(
    description=(
        "This function takes a single StockId and returns the sector of that stock."
        " Use this function to get the sector for a single stock when the user asks,"
        " for example, to compare a specific stock against a baseline of stocks in its "
        " sector, usually you will a this to get a sector and then use the sector filter"
        " on a standard universe like the S&P 500 to those filter stocks in that sector."
        " If you want to filter on a sector specifically mentioned by the client do not"
        " use this tool, look up the sector identifier instead."
        " If you want to look up the sector for a large number of stocks for the purposes"
        " of displaying them to the user or doing grouped calculations across multiple "
        " sectors, do not use this function, you should use the get_sector_for_stocks"
        " instead."
    ),
    category=ToolCategory.STOCK_GROUPS,
    tool_registry=default_tool_registry(),
    is_visible=True,
)
async def get_stock_sector(args: GetStockSectorInput, context: PlanRunContext) -> SectorID:
    """Returns the Sector of a stock

    Returns:
        SectorID: sector.
    """

    rows = await get_stock_metadata_rows(gbi_ids=[args.stock_id.gbi_id])
    if not rows:
        return SectorID(sec_id=-1, sec_name="Unknown Sector")

    return SectorID(sec_id=rows[0]["gics1_sector"], sec_name=rows[0]["gics1_name"])


@cached(cache=TTLCache(maxsize=1, ttl=ONE_HOUR), lock=Lock())
def get_all_gics_classifications() -> Dict[str, Dict]:
    """
    We return all valid sectors, industry groups, industries, and sub-industries included
    in the GICS classification, including "No Sector", hence the special case id = -1
    """

    db = get_psql()
    sql = """SELECT id as sector_id, name as sector_name
            FROM    GIC_SECTOR
            WHERE   active = TRUE
            ORDER BY sector_id, length(id::text)"""
    rows = db.generic_read(sql)
    return {str(r["sector_id"]): r for r in rows}


@cached(cache=TTLCache(maxsize=1, ttl=ONE_HOUR), lock=Lock())
def get_all_top_level_sectors() -> Dict[str, str]:
    """
    We return all sectors, including "No Sector", hence the special case id = -1
    """

    db = get_psql()
    sql = """SELECT id as sector_id, name as sector_name
            FROM    GIC_SECTOR
            WHERE   parent_id = 0 AND active = TRUE
            ORDER BY sector_id, length(id::text)"""
    rows = db.generic_read(sql)
    return {str(r["sector_id"]): str(r["sector_name"]) for r in rows}


def get_child_sectors_from_parent(parent_sector_ids: List[str]) -> Dict[str, str]:
    """
    We return child sectors of parent_sector_ids
    """

    db = get_psql()
    sql = """SELECT id as sector_id, name as sector_name
            FROM    GIC_SECTOR
            WHERE   parent_id = ANY(%(parent_sector_ids)s) and active = TRUE
            ORDER BY sector_id, length(id::text)"""
    rows = db.generic_read(sql, params={"parent_sector_ids": parent_sector_ids})
    return {str(r["sector_id"]): str(r["sector_name"]) for r in rows}


ALL_GICS_STR = ""


def get_all_gics_str() -> str:
    """
    Returns a formatted string of all sectors and sub-sectors
    e.g.
    Sectors:
    Energy
    Materials
    ...

    Industry Groups:
    Oil & Gas Drilling
    Oil & Gas Equipment & Services
    Integrated Oil & Gas
    ...
    """

    global ALL_GICS_STR
    if ALL_GICS_STR:
        return ALL_GICS_STR

    ALL_GICS_STR = "Sectors:\n"
    sectors = get_all_top_level_sectors()
    ALL_GICS_STR += "\n".join(sectors.values())
    ALL_GICS_STR += "\n\nIndustry Groups:\n"
    industry_groups = get_child_sectors_from_parent(list(sectors.keys()))
    ALL_GICS_STR += "\n".join(industry_groups.values())
    ALL_GICS_STR += "\n\nIndustries:\n"
    industries = get_child_sectors_from_parent(list(industry_groups.keys()))
    ALL_GICS_STR += "\n".join(industries.values())
    ALL_GICS_STR += "\n\nSub-Industries:\n"
    sub_industries = get_child_sectors_from_parent(list(industries.keys()))
    ALL_GICS_STR += "\n".join(sub_industries.values())

    return ALL_GICS_STR


class SectorIdentifierLookupInput(ToolArgs):
    sector_name: str


SECTOR_LOOKUP_PROMPT = Prompt(
    name="SECTOR_LOOKUP_PROMPT",
    template="""
Your task is to, in a finance context,
identify which (if any) of a provided list of GICS economic classifications
corresponds to a particular provided reference.
If there is an exact match, or one with a strong semantic overlap,
return the sector_id of the match,
otherwise return -1 to indicate "No Sector".
You should always first attempt an exact, word-for-word match, where the provided
reference text matches exactly to the name of the classification (sector_name).
Only attempt to find the matching
classification name which is most semantically similar to the provided
reference text (sector_name) if you are not able to find an exact match.

Here are the Sectors in json format:
---
{lookup_list}
---

And here is the user provided reference text you are trying to match: '{text_input}'

Now output the match from the list if you have found it and also provide a short reason

Make sure to return this in JSON.
ONLY RETURN IN JSON. DO NOT RETURN NON-JSON.
Do not return anything else.
Return in this format: {{"correct_sector_id":"", "reason":""}}
""",
)


@tool(
    description=f"""
This function takes a string like 'Healthcare' which
refers to a GICS industry classification and converts it to an identifier.
You should make the input string specific to what the customer is looking for.
For example, if the client asks for `car companies`, the input string should be `Automobiles`
You must ONLY use this look up and the corresponding sector filter tool if the client specifically
asks for one of these classifications directly. Sometimes the client may use the word `sector` to
refer to things which do not actually correspond to GICS sectors, in such a case you will NOT use this tool.
You should only use this tool if what the client says closely corresponds to a sector on your list, if
it does not, you must NOT pick a related sector, instead you should use the
`filter_stocks_by_product_or_service` tool.
Sometimes the client may use words such as `institutions`, `firms`, `companies`, or `stocks` to refer
to a particular category or selection of stocks which may actually refer to a sector.
For example, if someone mentions 'high bond yield sector', 'high bond yield' does not correspond to
anything in the above list of sectors and you must NOT use this sector tool under any circumstances!
If the client asks to filter stocks by something more specific than an industry/sector, you must use the
filter_product_or_service tool (for simple requests involving a clear class of product or services)
or filter_by_profile_tool (for more complex stock filtering).
This function supports the following GICS classifications:

{get_all_gics_str()}
""",
    category=ToolCategory.STOCK_GROUPS,
    tool_registry=default_tool_registry(),
)
async def sector_identifier_lookup(
    args: SectorIdentifierLookupInput, context: PlanRunContext
) -> SectorID:
    """
    Returns integer identifier best matching the input text, or None if not a match
    """
    logger = get_prefect_logger(__name__)
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O_MINI)

    all_sectors = get_all_gics_classifications()
    lookup_prompt = SECTOR_LOOKUP_PROMPT.format(
        lookup_list=json.dumps(all_sectors, sort_keys=True, indent=4), text_input=args.sector_name
    )

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=lookup_prompt,
        sys_prompt=NO_PROMPT,
        output_json=True,
    )

    logger.info(f"'{args.sector_name=}' '{result=}'")
    res_obj = json.loads(repair_json_if_needed(result))

    found_sector = str(res_obj.get("correct_sector_id", "-1"))
    if found_sector not in found_sector or "-1" == found_sector:
        raise ValueError(f"Could not map text '{args.sector_name}' to a GICS sector")

    sector = all_sectors[found_sector]

    await tool_log(
        log=f"Interpreting '{args.sector_name}' as GICS Sector: {sector.get('sector_name')}",
        context=context,
    )
    # or should this return the whole sector dict id + name
    # (maybe we should add descriptions of each sector as well?
    return SectorID(sec_id=int(sector["sector_id"]), sec_name=sector["sector_name"])


# can't use cache decorator because even though we
# need user id to qry svc it doesn't change the answer
DEFAULT_STOCK_LIST: List[StockID] = []


async def get_default_stock_list(user_id: str) -> List[StockID]:
    """
    Returns a default stock list (sp500 for now)
    # int he future this might be user aware
    """

    global DEFAULT_STOCK_LIST
    if DEFAULT_STOCK_LIST:
        return DEFAULT_STOCK_LIST

    SPY_SP500_GBI_ID = 10076  # same on dev & prod
    resp = await sec_meta_svc_client.get_etf_holdings(SPY_SP500_GBI_ID, user_id)

    gbi_ids = [stock.gbi_id for stock in resp.etf_universe_holdings[0].holdings.weighted_securities]
    # shouldn't have to do this but there was a bug on dev where dupes were being returned
    gbi_ids = list(set(gbi_ids))
    stocks = await StockID.from_gbi_id_list(gbi_ids)

    gbi_ids = list(set(gbi_ids))
    if gbi_ids:
        DEFAULT_STOCK_LIST = stocks
    return DEFAULT_STOCK_LIST


class SectorFilterInput(ToolArgs):
    sector_id: SectorID
    stock_ids: Optional[List[StockID]] = None
    etf_min_sector_threshold: Optional[float] = 25.0


@tool(
    description="""
This function was renamed to gics_sector_industry_filter
""",
    category=ToolCategory.STOCK_FILTERS,
    tool_registry=default_tool_registry(),
    enabled=False,
)
async def sector_filter(args: SectorFilterInput, context: PlanRunContext) -> List[StockID]:
    return await gics_sector_industry_filter(args, context)  # type:ignore


@tool(
    description="""
This function takes a sector id integer and an optional list of stocks
and filters the list to only those stocks whose sector matches the sector_id
If no stocks are passed in, a suitable default list such as S&P500 will be used
Returns a list of stock_ids filtered by sector.
You must call the sector_identifier_lookup tool as a separate step before this tool to
get a correct sector identifier You must use this tool (after calling sector_identifier_lookup
as a separate step!) if the client asks for filtering by specific sector, NEVER use the
get_sector_for_stocks tool for filtering.
Sometimes the client may use the word `sector` to
refer to things which do not correspond to GICS sectors, in such a case you will NOT use this tool.
The client may use words such as `institutions`, `firms`, `companies`, or `stocks` to refer to
sectors and this tool should be used filter stocks.
If client mentions specific stock names, be sure to include them in the list of stocks.
This tool can be used to filter both stocks and ETFs, only try to filter for ETFs if the user specifically
mentions ETFs or funds.
For ETFs, an optional sector threshold can be passed in which specifies the amount of holdings
for an ETF that should match to the specified sector.
This tool filters to a particular sector, all of the output stocks will be in the sector. If the
client is interested in filtering out a particular sector, first call this tool with the sector
you want to filter out, and then use the diff_list tool, subtracting the original stock list from
this new list. Listen carefully, you must not use this tool only and assume you are 'filtering out',
you will do exactly the opposite of what the client wants, the plan will fail, and you will be
fired!!!!
""",
    category=ToolCategory.STOCK_FILTERS,
    tool_registry=default_tool_registry(),
)
async def gics_sector_industry_filter(
    args: SectorFilterInput, context: PlanRunContext
) -> List[StockID]:
    """
    Returns a sector-filtered list of gbi_ids
    """

    logger = get_prefect_logger(__name__)

    stock_ids = args.stock_ids
    if stock_ids == []:
        # degenerate case should i log or throw?
        await tool_log(log="No stocks left to filter by sector", context=context)
        return []

    if stock_ids is None:
        stock_ids = await get_default_stock_list(user_id=context.user_id)

    if not args.etf_min_sector_threshold:
        args.etf_min_sector_threshold = 25.0  # set default

    sector_id_str = str(args.sector_id.sec_id)
    included_gbi_ids = set()

    db = get_psql()

    # First, take out all the gbi_ids that are ETFs for a separate sector filter
    etf_sql = """
        SELECT ms.gbi_security_id
        FROM master_security ms
        WHERE ms.security_type LIKE %(etf_type)s
        AND ms.gbi_security_id = ANY(%(stock_ids)s)
    """
    etf_rows = db.generic_read(
        etf_sql,
        params={"stock_ids": [stock.gbi_id for stock in stock_ids], "etf_type": "%ETF%"},
    )
    etf_stock_ids = {etf["gbi_security_id"] for etf in etf_rows}

    if etf_stock_ids:
        etf_sectors = await sec_meta_svc_client.get_all_etf_sectors(
            gbi_ids=list(etf_stock_ids),
            user_id=context.user_id,
        )

        for etf in etf_sectors.breakdowns:
            sector_weight = 0.0
            if len(sector_id_str) == 2:
                sector_weight = etf.sectors[int(sector_id_str)]
            elif len(sector_id_str) == 4:
                sector_weight = etf.industry_groups[int(sector_id_str)]
            elif len(sector_id_str) == 6:
                sector_weight = etf.industries[int(sector_id_str)]
            else:
                sector_weight = etf.sub_industries[int(sector_id_str)]
            if sector_weight >= (args.etf_min_sector_threshold * 0.01):
                included_gbi_ids.add(etf.gbi_id)

    # Add a fallback here if sector ID is too specific?
    sector_filter_clause = "ms.gics = %(sector_id)s"
    if len(sector_id_str) == 2:
        sector_filter_clause = "CAST(SUBSTRING(CAST(ms.gics AS TEXT), 1, 2) AS INT) = %(sector_id)s"
    elif len(sector_id_str) == 4:
        sector_filter_clause = "CAST(SUBSTRING(CAST(ms.gics AS TEXT), 1, 4) AS INT) = %(sector_id)s"
    elif len(sector_id_str) == 6:
        sector_filter_clause = "CAST(SUBSTRING(CAST(ms.gics AS TEXT), 1, 6) AS INT) = %(sector_id)s"

    sql = f"""
    SELECT ms.gbi_security_id
    FROM master_security ms
    WHERE {sector_filter_clause}
    AND ms.gbi_security_id = ANY(%(stock_ids)s)
    """

    rows = db.generic_read(
        sql,
        params={
            "stock_ids": [stock.gbi_id for stock in stock_ids if stock.gbi_id not in etf_stock_ids],
            "sector_id": args.sector_id.sec_id,
        },
    )
    for row in rows:
        included_gbi_ids.add(row["gbi_security_id"])
    stock_list = [stock for stock in stock_ids if stock.gbi_id in included_gbi_ids]
    if not stock_list:
        raise EmptyOutputError(message="Stock Sector filter resulted in an empty list of stocks")
    await tool_log(
        log=f"Filtered {len(stock_ids)} stocks by sector down to {len(stock_list)}", context=context
    )

    try:  # since everything associated with diffing is optional, put in try/except
        # we need to add the task id to all runs, including the first one, so we can track changes
        # Update mode
        if context.task_id:
            stock_list = add_task_id_to_stocks_history(stock_list, context.task_id)
            if context.diff_info is not None:
                prev_run_info = await get_prev_run_info(context, "sector_filter")
                if prev_run_info is not None:
                    prev_input = SectorFilterInput.model_validate_json(prev_run_info.inputs_str)
                    prev_output: List[StockID] = prev_run_info.output  # type:ignore
                    # corner case here where S&P 500 change causes output to change, but not going to
                    # bother with it on first pass
                    if args.stock_ids and prev_input.stock_ids:
                        # we only care about stocks that were inputs for both
                        shared_inputs = set(prev_input.stock_ids) & set(args.stock_ids)
                    else:
                        shared_inputs = set(await get_default_stock_list(user_id=context.user_id))
                    curr_stock_set = set(stock_list)
                    prev_stock_set = set(prev_output)
                    added_stocks = (curr_stock_set - prev_stock_set) & shared_inputs
                    removed_stocks = (prev_stock_set - curr_stock_set) & shared_inputs
                    context.diff_info[context.task_id] = {
                        "added": {
                            added_stock: SECTOR_ADD_STOCK_DIFF.format(
                                company=added_stock.company_name, sector=args.sector_id.sec_name
                            )
                            for added_stock in added_stocks
                        },
                        "removed": {
                            removed_stock: SECTOR_REMOVE_STOCK_DIFF.format(
                                company=removed_stock.company_name, sector=args.sector_id.sec_name
                            )
                            for removed_stock in removed_stocks
                        },
                    }

    except Exception as e:
        logger.exception(f"Error creating diff info from previous run: {e}")
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to get previous run info or getting default stock list",
        )

    return stock_list
