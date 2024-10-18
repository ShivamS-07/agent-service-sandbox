import json
from bisect import bisect
from collections import defaultdict
from typing import Dict, List

from agent_service.GPT.constants import GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_types.stock import StockID
from agent_service.io_types.stock_groups import StockGroup, StockGroups
from agent_service.io_types.table import StockTable
from agent_service.planner.errors import BadInputError
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.stock_groups.constants import COLUMN, THRESHOLDS
from agent_service.tools.stock_groups.prompts import (
    CREATE_GROUP_MAIN_PROMPT,
    CREATE_STOCK_GROUPS_DESCRIPTION,
)
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.string_utils import clean_to_json_if_needed


class CreateStockGroupsFromTableInput(ToolArgs):
    table: StockTable
    definition: str
    header: str


@tool(description=CREATE_STOCK_GROUPS_DESCRIPTION, category=ToolCategory.STOCK, enabled=True)
async def create_stock_groups_from_table(
    args: CreateStockGroupsFromTableInput, context: PlanRunContext
) -> StockGroups:
    logger = get_prefect_logger(__name__)
    labels = [column.metadata.label for column in args.table.columns]

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)

    main_prompt = CREATE_GROUP_MAIN_PROMPT.format(definition=args.definition, labels=labels)
    result = await llm.do_chat_w_sys_prompt(
        main_prompt, NO_PROMPT
    )  # no json mode since outputing list
    group_definitions_json = json.loads(clean_to_json_if_needed(result))
    stock_column = args.table.get_stock_column()
    if stock_column is None:
        raise BadInputError("No stock column in input table")
    stock_list: List[StockID] = stock_column.data  # type:ignore
    if len(stock_list) < 2:
        raise BadInputError("Need at least 2 stocks in table to group")
    group_dict: Dict[str, List[StockID]] = {"": stock_list}

    for group_split in group_definitions_json:
        thresholds = None
        try:
            column = args.table.columns[labels.index(group_split[COLUMN])]
            if THRESHOLDS in group_split:
                thresholds = group_split[THRESHOLDS]
            column_lookup = dict(zip(stock_list, column.data))
        except (IndexError, KeyError):
            logger.warning(f"column for stock group not found in table, {group_split}")
            continue

        new_group_dict = defaultdict(list)

        for curr_label, curr_stocks in group_dict.items():
            if thresholds is None:  # str-based
                for stock in curr_stocks:
                    new_label = str(column_lookup[stock])
                    if curr_label:
                        new_label = curr_label + ", " + new_label
                    new_group_dict[new_label].append(stock)
            else:  # numerical
                for stock in curr_stocks:
                    index = bisect(thresholds, column_lookup[stock])
                    if index == 0:
                        new_label = str(column.metadata.label) + " < " + str(thresholds[0])
                    elif index == len(thresholds):
                        new_label = str(column.metadata.label) + " >= " + str(thresholds[-1])
                    else:
                        new_label = (
                            str(thresholds[index - 1])
                            + " <= "
                            + str(column.metadata.label)
                            + " < "
                            + str(thresholds[index])
                        )
                    if curr_label:
                        new_label = curr_label + ", " + new_label
                    new_group_dict[new_label].append(stock)

        group_dict = new_group_dict

    stock_groups = []
    for label, stocks in group_dict.items():
        stock_groups.append(StockGroup(name=label, stocks=stocks))
    return StockGroups(header=args.header, stock_groups=stock_groups)
