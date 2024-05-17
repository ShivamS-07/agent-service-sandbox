from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql


class StatisticsIdentifierLookupInput(ToolArgs):
    # name of the statistic to lookup
    statistic_name: str


@tool(
    description=(
        "This function takes a string (Churn low, Market Capitalization, Coppock Curve, e.g.)"
        "which refers to a statistic, and converts it to a string identifier"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
    is_visible=False,
)
async def statistic_identifier_lookup(
    args: StatisticsIdentifierLookupInput, context: PlanRunContext
) -> str:
    """Returns the string identifier of a statistic given its name (Churn low, Market Capitalization, e.g.)

    This function performs word similarity name match to find the statistic's identifier.


    Args:
        args (StatisticsIdentifierLookupInput): The input arguments for the statistic lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        str: The integer identifier of the statistic.
    """
    db = get_psql()
    # TODO :
    # 1. Add more filtering or new column (agent_supported) to table
    # 2. implement solar solution

    # Word similarity name match
    sql = """
    SELECT id FROM public.features feat
    WHERE feat.data_provider = 'SPIQ'
    ORDER BY word_similarity(lower(feat.name), lower(%s)) DESC
    LIMIT 1
    """
    rows = db.generic_read(sql, [args.statistic_name])
    if rows:
        return rows[0]["id"]

    raise ValueError(f"Could not find the stock {args.statistic_name}")
