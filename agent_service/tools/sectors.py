import json
from threading import Lock
from typing import Dict, List, Optional

from cachetools import TTLCache, cached

from agent_service.external import sec_meta_svc_client
from agent_service.GPT.constants import HAIKU, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import Text
from agent_service.planner.errors import NonRetriableError
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.postgres import get_psql
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
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
        t = Text(val=f"Sector: {self.sec_name}")
        return await t.to_rich_output(pg=pg, title=title)


class SectorIdentifierLookupInput(ToolArgs):
    sector_name: str


SECTOR_LOOKUP_PROMPT = Prompt(
    name="SECTOR_LOOKUP_PROMPT",
    template="""
Your task is to, in a finance context,
identify which (if any) of a provided list of GICS economic sectors
corresponds to a particular provided reference to a sector.
If there is an exact match, or one with a strong semantic overlap,
return the sector_id of the match,
otherwise return -1 to indicate "No Sector".

Here are the Sectors in json format:
---
{lookup_list}
---

And here is the user provided sector text you are trying to match: '{text_input}'

Now output the match from the list if you have found it and also provide a short reason

Make sure to return this in JSON.
ONLY RETURN IN JSON. DO NOT RETURN NON-JSON.
Do not return anything else.
Return in this format: {{"correct_sector_id":"", "reason":""}}
""",
)


@tool(
    description="""
This function takes a string like 'Healthcare' which
refers to a GICS economic sector and converts it to an identifier
This function supports the following economic sectors:
Energy
Materials
Industrials
Consumer Discretionary
Consumer Staples
Health Care
Financials
Information Technology
Communication Services
Utilities
Real Estate
You must ONLY use this look up and the corresponding sector filter tool if the user specifically
asks for one of these sectors directly. If the user asks for something else, including subsectors
of these sectors, you must use the filter_product_or_service tool (for simple requests involving
a clear class of product or services) or filter_by_profile_tool (for more complex stock filtering).
""",
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
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
    llm = GPT(context=gpt_context, model=HAIKU)

    all_sectors = get_all_sectors()
    lookup_prompt = SECTOR_LOOKUP_PROMPT.format(
        lookup_list=json.dumps(all_sectors, indent=4), text_input=args.sector_name
    )

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=lookup_prompt,
        sys_prompt=NO_PROMPT,
        output_json=True,
    )

    logger.info(f"'{args.sector_name=}' '{result=}'")
    res_obj = json.loads(result)

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


@cached(cache=TTLCache(maxsize=1, ttl=ONE_HOUR), lock=Lock())
def get_all_sectors() -> Dict[str, Dict]:
    """
    We return all sectors, including "No Sector", hence the special case id = -1
    """

    db = get_psql()
    sql = """SELECT id as sector_id, name as sector_name
            FROM    GIC_SECTOR
            WHERE   parent_id = 0 or id = -1"""
    rows = db.generic_read(sql)
    return {str(r["sector_id"]): r for r in rows}


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


@tool(
    description="""
This function takes a sector id integer and an optional list of stocks
and filters the list to only those stocks whose sector matches the sector_id
If no stocks are passed in, a suitable default list such as S&P500 will be used
Returns a list of stock_ids filtered by sector
""",
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def sector_filter(args: SectorFilterInput, context: PlanRunContext) -> List[StockID]:
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

    db = get_psql()

    sql = """
    SELECT ms.gbi_security_id
    FROM master_security ms
    WHERE CAST(SUBSTRING(CAST(ms.gics AS TEXT), 1, 2) AS INT) = %(sector_id)s
    AND ms.gbi_security_id = ANY(%(stock_ids)s)
    """

    # if we ever need to filter one subsectors use these:
    # cast(substring(cast(ms.gics as text), 1, 4) as int) as gics_industry_group_id,
    # cast(substring(cast(ms.gics as text), 1, 6) as int) as gics_industry_id,
    # ms.gics as gics_subindustry_id,

    rows = db.generic_read(
        sql,
        params={
            "stock_ids": [stock.gbi_id for stock in stock_ids],
            "sector_id": args.sector_id.sec_id,
        },
    )
    included_gbi_ids = {row["gbi_security_id"] for row in rows}
    stock_list = [stock for stock in stock_ids if stock.gbi_id in included_gbi_ids]
    if not stock_list:
        raise NonRetriableError(message="Stock Sector filter resulted in an empty list of stocks")
    await tool_log(
        log=f"Filtered {len(stock_ids)} stocks by sector down to {len(stock_list)}", context=context
    )

    try:  # since everything associated with diffing is optional, put in try/except
        # we need to add the task id to all runs, including the first one, so we can track changes
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
        logger.warning(f"Error creating diff info from previous run: {e}")

    return stock_list
