import datetime
from typing import List, Optional

import pandas as pd

from agent_service.io_type_utils import IOType
from agent_service.io_types.graph import LineGraph
from agent_service.io_types.table import Table
from agent_service.io_types.text import StockNewsDevelopmentText, Text
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.lists import _unionize_lists
from agent_service.types import PlanRunContext
from agent_service.utils.output_utils.output_construction import PreparedOutput

_TEST_REGISTRY = ToolRegistry()


def get_test_registry() -> ToolRegistry:
    class SummarizeTextInput(ToolArgs):
        texts: List[Text]

    @tool(
        description=(
            "This function takes a list of texts and uses an LLM to summarize them into a single text "
            "based on the instructions provided by the user in their input. Note: before you run this"
            " function you must make sure to apply all relevant filters on the texts, do not use "
            " this function to filter large quantities of text"
        ),
        category=ToolCategory.TEXT_WRITER,
        tool_registry=_TEST_REGISTRY,
    )
    async def summarize_texts(args: SummarizeTextInput, context: PlanRunContext) -> Text:
        return Text(id="1", val="A summarized text!")

    class FilterTextsByTopicInput(ToolArgs):
        topic: str
        texts: List[Text]

    @tool(
        description=(
            "This function takes a topic and list of texts and uses an LLM to filter the list to only those"
            " that are relevant to the provided topic. Can be applied to news, earnings, SEC filings, and any"
            " other text. "
            " It is better to call this function once with a complex topic with many ideas than to call this"
            " function many times with smaller topics. Use filter_items_by_topic if you have things other "
            "than texts that you want to filter"
        ),
        category=ToolCategory.STOCK_FILTERS,
        tool_registry=_TEST_REGISTRY,
    )
    async def filter_texts_by_topic(
        args: FilterTextsByTopicInput, context: PlanRunContext
    ) -> List[Text]:
        return []

    class GetNewsDevelopmentsAboutCompaniesInput(ToolArgs):
        stock_ids: List[int]
        start_date: datetime.date
        end_date: datetime.date = datetime.date.today()

    @tool(
        description=(
            "This function calls an internal API which provides all the news developments with articles "
            "between the start date and the end date that are relevant to the provided list of stocks,"
            "the output is a list of list of news development identifiers, each internal list corresponds"
            " to an input company"
        ),
        category=ToolCategory.NEWS,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_news_developments_about_companies(
        args: GetNewsDevelopmentsAboutCompaniesInput, context: PlanRunContext
    ) -> List[List[StockNewsDevelopmentText]]:
        return [[StockNewsDevelopmentText(id="1")]]

    class CollapseListsInput(ToolArgs):
        lists_of_lists: List[List[IOType]]

    @tool(
        description="This function collapses a list of lists into a list",
        category=ToolCategory.LIST,
        tool_registry=_TEST_REGISTRY,
    )
    async def collapse_lists(args: CollapseListsInput, context: PlanRunContext) -> List[IOType]:
        return []

    class CombineListsInput(ToolArgs):
        list1: List[IOType]
        list2: List[IOType]

    @tool(
        description=(
            "This function forms a single deduplicated list from the elements of two lists. "
            " For example, [1, 2, 3] and [3, 4, 5] would add to [1, 2, 3, 4, 5]."
            " This is particularly useful if you created two lists of stocks or texts and want to"
            " put them together into a single list"
            " This is equivalent to `boolean OR` or `Union` logic, if you only want elements in "
            " both lists, use intersect_lists"
            " This is the ONLY way to combine lists, you must NEVER, EVER use the + operator in the plan"
            " Note that like all other tools, this tool must be called as as a separate step of the plan!"
            " Note that the output type is really a union of the input list types."
        ),
        category=ToolCategory.LIST,
        tool_registry=_TEST_REGISTRY,
        is_visible=False,
        output_type_transformation=_unionize_lists,
    )
    async def add_lists(args: CombineListsInput, context: PlanRunContext) -> List[IOType]:
        result = args.list1 + args.list2
        return result

    class GetDateFromDateStrInput(ToolArgs):
        time_str: str

    @tool(
        description=(
            "This function takes a string which refers to a time,"
            " either absolute or relative to the current time, and converts it to a Python date"
        ),
        category=ToolCategory.DATES,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_date_from_date_str(
        args: GetDateFromDateStrInput, context: PlanRunContext
    ) -> datetime.date:
        return datetime.date.today()

    class StockIdentifierLookupInput(ToolArgs):
        stock_name: str

    @tool(
        description=(
            "This function takes a string which refers to a stock, and converts it to an integer identifier "
            "You should use the multi version of this function in any case where more than one stock lookup is needed"
        ),
        category=ToolCategory.STOCK,
        tool_registry=_TEST_REGISTRY,
    )
    async def stock_identifier_lookup(
        args: StockIdentifierLookupInput, context: PlanRunContext
    ) -> int:
        return 0

    class StockIdentifierLookupMultiInput(ToolArgs):
        stock_names: List[str]

    @tool(
        description=(
            "This function takes a list of strings each of which refers to a stock, "
            "and converts them to a list of integer identifiers"
        ),
        category=ToolCategory.STOCK,
        tool_registry=_TEST_REGISTRY,
    )
    async def stock_identifier_lookup_multi(
        args: StockIdentifierLookupMultiInput, context: PlanRunContext
    ) -> List[int]:
        return [0]

    # Earnings summary test

    class GetUserPortfolioStocksInput(ToolArgs):
        portfolio_name: List[str] = []

    @tool(
        description=(
            "This function returns a list of stock identifiers for all stocks in the provided "
            "users portfolios or all portfolios if portfolio_name is an empty list (the default value)"
        ),
        category=ToolCategory.PORTFOLIO,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_user_portfolio_stocks(
        args: GetUserPortfolioStocksInput, context: PlanRunContext
    ) -> List[int]:
        return []

    class GetEarningsImpactsInput(ToolArgs):
        impacted_stock_identifiers: List[int]

    @tool(
        description=(
            "This function returns a list of list of stock identifiers, each list of stocks corresponds"
            " to the stocks whose earnings calls are likely to have an impact on the"
            " that stock's performance"
        ),
        category=ToolCategory.STOCK,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_earnings_impacts(
        args: GetEarningsImpactsInput, context: PlanRunContext
    ) -> List[List[int]]:
        return [[]]

    class GetEarningsCallSummaries(ToolArgs):
        stock_ids: List[int]
        start_date: Optional[datetime.date] = None
        end_date: Optional[datetime.date] = None

    @tool(
        description=(
            "This returns a list of lists of earnings call summary texts, each inner list corresponds to all the"
            " earnings calls for the corresponding stock that were published between start_date and end_date. "
            "start_date or end_date being None indicates the range is unbounded"
        ),
        category=ToolCategory.EARNINGS,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_earnings_call_summaries(
        args: GetEarningsCallSummaries, context: PlanRunContext
    ) -> List[List[Text]]:
        return [[]]

    # profit margin example

    class GetNamesOfSingleStockInput(ToolArgs):
        stock_id: int

    @tool(
        description="Gets the name of the stock indicated by the stock_id",
        category=ToolCategory.STOCK,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_name_of_single_stock(
        args: GetNamesOfSingleStockInput, context: PlanRunContext
    ) -> str:
        return ""

    class GetNamesOfStocksInput(ToolArgs):
        stock_ids: List[int]

    @tool(
        description="Gets the names of the stocks indicated by the stock_ids",
        category=ToolCategory.STOCK,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_names_of_stocks(
        args: GetNamesOfStocksInput, context: PlanRunContext
    ) -> List[str]:
        return []

    class GetElementFromListInput(ToolArgs):
        L: List[IOType]
        n: int

    @tool(
        description="Get the nth element of a list. You must use this instead of the Python indexing ([])",
        category=ToolCategory.LIST,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_element_from_list(
        args: GetElementFromListInput, context: PlanRunContext
    ) -> IOType:
        return args.L[args.n]

    class GetStatisticIdentifierInput(ToolArgs):
        statistic_reference: str

    @tool(
        description=(
            "This function takes a text reference to some statistic and converts it to an identifier"
            " which can be used to look it up in the database"
        ),
        category=ToolCategory.STATISTICS,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_statistic_identifier(
        args: GetStatisticIdentifierInput, context: PlanRunContext
    ) -> str:
        return ""

    class GetCompanyStatsOverDatesInput(ToolArgs):
        stock_ids: List[int]
        stock_labels: List[str]
        statistic_id: str
        start_date: datetime.date
        end_date: Optional[datetime.date] = None

    @tool(
        description=(
            "This function queries the database to get the specific values of the statistic"
            " referred to by statistic_id, for all the stocks in stock_ids, over the time"
            " range indicated, if end_date is not that means it is up to the present. It returns"
            " a StockTimeSeriesTable where the rows are stocks and the columns are the dates "
            " stock_labels should be human-readable names or tickers that will be shown instead of"
            " the identifiers if there is any visualization of this table, there must be as"
            " many labels as there are stock_ids"
        ),
        category=ToolCategory.STATISTICS,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_company_stats_over_dates(
        args: GetCompanyStatsOverDatesInput, context: PlanRunContext
    ) -> Table:
        return Table.from_df_and_cols(data=pd.DataFrame([[0]]), columns=[])

    class AverageTableByDateInput(ToolArgs):
        table: Table
        new_column_header: str

    @tool(
        description=(
            "This function collapses a time series table to a single column by taking"
            " the average (mean) score across all columns for each date, the resulting table"
            " has a single column with the provided new_column header"
        ),
        category=ToolCategory.TABLE,
        tool_registry=_TEST_REGISTRY,
    )
    async def average_table_by_date(
        args: AverageTableByDateInput, context: PlanRunContext
    ) -> Table:
        return Table.from_df_and_cols(data=pd.DataFrame([[0]]), columns=[])

    class ConcatTimeSeriesTableInput(ToolArgs):
        table1: Table
        table2: Table

    @tool(
        description=(
            "This function concatenates two compatible time series tables together, the resulting"
            " table has the same dates (the rows) and all the columns in both tables"
        ),
        category=ToolCategory.TABLE,
        tool_registry=_TEST_REGISTRY,
    )
    async def concat_time_series_table(
        args: ConcatTimeSeriesTableInput, context: PlanRunContext
    ) -> Table:
        return Table.from_df_and_cols(data=pd.DataFrame([[0]]), columns=[])

    class PlotLineGraphInput(ToolArgs):
        table: Table

    @tool(
        description=(
            "This function plots a Time series table, each column will become a line"
            " on the output graph, with the label corresponding to that column header"
        ),
        category=ToolCategory.OUTPUT,
        tool_registry=_TEST_REGISTRY,
    )
    async def PlotLineGraphInput(args: PlotLineGraphInput, context: PlanRunContext) -> LineGraph:
        return None

    # health care companies example

    class SectorLookupInput(ToolArgs):
        sector_ref: str

    @tool(
        description="This takes a text reference to a sector and converts it into a sector identifier",
        category=ToolCategory.STOCK,
        tool_registry=_TEST_REGISTRY,
    )
    async def sector_lookup(args: SectorLookupInput, context: PlanRunContext) -> str:
        return ""

    class GetStocksInSectorInput(ToolArgs):
        sector_id: str

    @tool(
        description=(
            "Given a sector_id produced by the sector_lookup function, this returns stock ids for all "
            "stocks in that sector"
        ),
        category=ToolCategory.STOCK,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_stocks_in_sector(
        args: GetStocksInSectorInput, context: PlanRunContext
    ) -> List[int]:
        return []

    class GetCurrentCompanyStatsInput(ToolArgs):
        stock_ids: List[int]
        statistic_id: str

    @tool(
        description="This queries a database to get the current value of a statistic for each of a set of stocks",
        category=ToolCategory.STATISTICS,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_current_company_stats(
        args: GetCurrentCompanyStatsInput, context: PlanRunContext
    ) -> List[float]:
        return []

    class FilterStocksByStatisticsInput(ToolArgs):
        stock_ids: List[int]
        statistic_values: List[float]
        threshold: float
        greater_than: bool

    @tool(
        description=(
            "This function filters a lists of stocks based on the value of some statistic, "
            "stock_ids and statistic_values are of the same length and are aligned, "
            " if greater_than is true, should return a list of stocks whose corresponding "
            " statistic is higher than the threshold, or a list of stocks below if greater_than is false"
        ),
        category=ToolCategory.STATISTICS,
        tool_registry=_TEST_REGISTRY,
    )
    async def filter_stocks_by_statistic(
        args: FilterStocksByStatisticsInput, context: PlanRunContext
    ) -> List[int]:
        return []

    class FilterItemsByTopicInput(ToolArgs):
        topic: str
        items: List[IOType]
        texts: List[Text]

    @tool(
        description=(
            "This function takes any list of items which has some corresponding associated texts"
            " and uses an LLM to filter to only those objects relevant to the provided topic."
        ),
        category=ToolCategory.STOCK_FILTERS,
        tool_registry=_TEST_REGISTRY,
    )
    async def filter_items_by_topic(
        args: FilterItemsByTopicInput, context: PlanRunContext
    ) -> List[IOType]:
        return []

    # Recession Theme

    class GetMacroeconomicTheme(ToolArgs):
        theme_reference: str

    @tool(
        description="This searches for an existing analysis of a macroeconomic theme and its effects "
        "on stocks. The search is based on a string reference to the theme. An theme identifier"
        " is returned",
        category=ToolCategory.THEME,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_macroeconomic_theme(args: GetMacroeconomicTheme, context: PlanRunContext) -> str:
        return ""

    class GetStocksAffectedByTheme(ToolArgs):
        theme_id: str
        positive: bool

    @tool(
        description=(
            "This gets a list of stocks (stock identifiers) that are either positively (if positive "
            "is True) or negatively affected (if positive is False) by the theme indicated by theme_id"
        ),
        category=ToolCategory.THEME,
        tool_registry=_TEST_REGISTRY,
    )
    async def get_stocks_affected_by_theme(
        args: GetStocksAffectedByTheme, context: PlanRunContext
    ) -> List[int]:
        return []

    class ConvertStockIdentifiersToTickers(ToolArgs):
        stock_ids: List[int]

    class OutputArgs(ToolArgs):
        object_to_output: IOType
        title: str

    @tool(
        description="",
        category=ToolCategory.OUTPUT,
        is_visible=False,
        is_output_tool=True,
        store_output=False,
        tool_registry=_TEST_REGISTRY,
    )
    async def prepare_output(args: OutputArgs, context: PlanRunContext) -> PreparedOutput:
        return PreparedOutput(val=args.object_to_output)

    return _TEST_REGISTRY
