import json
from threading import Lock
from typing import Dict, List, Optional

from cachetools import TTLCache, cached

from agent_service.external import sec_meta_svc_client
from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.postgres import get_psql
from agent_service.utils.prompt_utils import Prompt

ONE_HOUR = 60 * 60


class SectorIdentifierLookupInput(ToolArgs):
    sector_name: str


SECTOR_LOOKUP_PROMPT = Prompt(
    name="SECTOR_LOOKUP_PROMPT",
    template="""
Your task is to identify which (if any) of a provided list of GICS economic sectors
corresponds to a particular provided reference to a sector.
If there is an exact match, or one with a strong semantic overlap,
return the sector_id of the match,
otherwise return -1 to indicate "No Sector".
Do not return anything else.
Here is the list of Sectors:
---
{lookup_list}
---
And here is the user provided sector text you are trying to match: {text_input}.
Now output the match from the list if you have found it:
""",
)


@tool(
    description="""
This function takes a string like 'Healthcare' which
refers to a GICS economic sector and converts it to an integer identifier,
or -1 if no a matching sector is found.
""",
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def sector_identifier_lookup(
    args: SectorIdentifierLookupInput, context: PlanRunContext
) -> int:
    """
    Returns integer identifier best matching the input text, or None if not a match
    """
    gpt_context = create_gpt_context(
        GptJobType.AGENT_PLANNER, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=DEFAULT_CHEAP_MODEL)

    all_sectors = get_all_sectors()
    lookup_prompt = SECTOR_LOOKUP_PROMPT.format(
        lookup_list=json.dumps(all_sectors, indent=4), text_input=args.sector_name
    )

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=lookup_prompt,
        sys_prompt=NO_PROMPT,
    )

    if result not in all_sectors or "-1" == result:
        # TODO we should actually throw an error and use it to revise the plan
        return -1  # or should we return None?

    sector = all_sectors[result]

    # or should this return the whole sector dict id + name
    # (maybe we should add descriptions of each sector as well?
    return sector["sector_id"]


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
DEFAULT_STOCK_LIST: List[int] = []


async def get_default_stock_list(user_id: str) -> List[int]:
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
    if gbi_ids:
        DEFAULT_STOCK_LIST = gbi_ids
    return DEFAULT_STOCK_LIST


class SectorFilterInput(ToolArgs):
    sector_id: int
    stock_ids: Optional[List[int]] = None


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
async def sector_filter(args: SectorFilterInput, context: PlanRunContext) -> List[int]:
    """
    Returns a sector-filtered list of gbi_ids
    """

    stock_ids = args.stock_ids
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

    rows = db.generic_read(sql, params={"stock_ids": stock_ids, "sector_id": args.sector_id})
    return [r["gbi_security_id"] for r in rows]
