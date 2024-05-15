from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql


class FeatureIdentifierLookupInput(ToolArgs):
    # name of the feature/variable to lookup
    feature_str: str


@tool(
    description=(
        "This function takes a string (Churn low, Market Capitalization, Coppock Curve, e.g.)"
        "which refers to a feature, and converts it to a string identifier"
    ),
    category=ToolCategory.STOCK,
    tool_registry=ToolRegistry,
)
async def feature_identifier_lookup(
    args: FeatureIdentifierLookupInput, context: PlanRunContext
) -> str:
    """Returns the string identifier of a feature given its name (Churn low, Market Capitalization, Coppock Curve, e.g.)

    This function performs word similarity name match to find the feature's identifier.


    Args:
        args (FeatureIdentifierLookupInput): The input arguments for the feature lookup.
        context (PlanRunContext): The context of the plan run.

    Returns:
        str: The integer identifier of the feature.
    """
    db = get_psql()
    # TODO :
    # 1. Add more filtering or new column (agent_supported) to table
    # 2. implement solar solution

    # Word similarity name match
    sql = """
    SELECT id FROM public.features feat
    ORDER BY word_similarity(lower(feat.name), lower(%s)) DESC
    LIMIT 1
    """
    rows = db.generic_read(sql, [args.feature_str])
    if rows:
        return rows[0]["id"]

    raise ValueError(f"Could not find the stock {args.feature_str}")
