import datetime
import unittest
import warnings
from typing import Any, List, Optional, Type, Union
from unittest import IsolatedAsyncioTestCase, TestCase

import pandas as pd

from agent_service.endpoints.models import AgentMetadata
from agent_service.GPT.requests import set_use_global_stub
from agent_service.io_type_utils import IOType
from agent_service.io_types.graph import LineGraph
from agent_service.io_types.table import Table
from agent_service.io_types.text import StockNewsDevelopmentText, Text
from agent_service.planner.executor import run_execution_plan_local
from agent_service.planner.planner import Planner
from agent_service.planner.planner_types import (
    ExecutionPlan,
    ParsedStep,
    ToolExecutionNode,
    Variable,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.logs import init_stdout_logging
from agent_service.utils.postgres import DEFAULT_AGENT_NAME, get_psql


def get_test_registry() -> Type[ToolRegistry]:
    class TestRegistry(ToolRegistry):
        pass

    # Stock news summary test

    class SummarizeTextInput(ToolArgs):
        texts: List[Text]

    @tool(
        description=(
            "This function takes a list of texts and uses an LLM to summarize them into a single text "
            "based on the instructions provided by the user in their input. Note: before you run this"
            " function you must make sure to apply all relevant filters on the texts, do not use "
            " this function to filter large quantities of text"
        ),
        category=ToolCategory.LLM_ANALYSIS,
        tool_registry=TestRegistry,
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
        category=ToolCategory.LLM_ANALYSIS,
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
    )
    async def collapse_lists(args: CollapseListsInput, context: PlanRunContext) -> List[IOType]:
        return []

    class GetDateFromDateStrInput(ToolArgs):
        time_str: str

    @tool(
        description=(
            "This function takes a string which refers to a time,"
            " either absolute or relative to the current time, and converts it to a Python date"
        ),
        category=ToolCategory.DATES,
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        category=ToolCategory.USER,
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
    )
    async def PlotLineGraphInput(args: PlotLineGraphInput, context: PlanRunContext) -> LineGraph:
        return None

    # health care companies example

    class SectorLookupInput(ToolArgs):
        sector_ref: str

    @tool(
        description="This takes a text reference to a sector and converts it into a sector identifier",
        category=ToolCategory.STOCK,
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        category=ToolCategory.LLM_ANALYSIS,
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
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
        tool_registry=TestRegistry,
    )
    async def get_stocks_affected_by_theme(
        args: GetStocksAffectedByTheme, context: PlanRunContext
    ) -> List[int]:
        return []

    class ConvertStockIdentifiersToTickers(ToolArgs):
        stock_ids: List[int]

    return TestRegistry


class TestPlans(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_use_global_stub(False)
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*<ssl.SSLSocket.*>"
        )
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*<socket.socket.*>"
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="The loop argument is deprecated since Python 3.8, and scheduled for removal in Python 3.10",  # noqa
        )

        init_stdout_logging()

    def setUp(self) -> None:
        self.tool_registry = get_test_registry()

    async def test_tool_registry_works(self) -> None:
        get_test_registry()

    @unittest.skip("Takes too long to run")
    async def test_planner(self) -> None:
        input_text = (
            "Can you give me a single summary of news published in the last month "
            "about machine learning at Meta, Apple, and Microsoft?"
        )
        # input_text = (
        #     "I need a summary of all the earnings calls from yesterday of companies"
        #     "that might impact stocks in my portfolio"
        # )
        # input_text = (
        #     "I want to see the profit margins of Microsoft for the last year graphed"
        #     "against the average of its peers."
        # )
        # input_text = (
        #     "We are looking for low-to-mid cap, high growth healthcare companies"
        #     " with innovative technologies that can become the standard of care for"
        #     " unmet medical needs"
        # )
        # input_text = (
        #     "I want to find good stocks to short so that if a recession happens I'm protected."
        # )
        user_message = Message(message=input_text, is_user_message=True)
        chat_context = ChatContext(messages=[user_message])
        planner = Planner("", tool_registry=self.tool_registry)
        await planner.create_initial_plan(chat_context)

    async def test_executor(self):
        plan = ExecutionPlan(
            nodes=[
                ToolExecutionNode(
                    tool_name="get_date_from_date_str",
                    args={"time_str": "1 month ago"},
                    description='Convert "1 month ago" to a date to use as the start date for news search',
                    output_variable_name="start_date",
                    is_output_node=False,
                ),
                ToolExecutionNode(
                    tool_name="stock_identifier_lookup_multi",
                    args={"stock_names": ["Meta", "Apple", "Microsoft"]},
                    description="Convert company names to stock identifiers",
                    output_variable_name="stock_ids",
                    is_output_node=False,
                ),
                ToolExecutionNode(
                    tool_name="get_news_developments_about_companies",
                    args={
                        "stock_ids": Variable(var_name="stock_ids"),
                        "start_date": Variable(var_name="start_date"),
                    },
                    description="Get news developments for Meta, Apple, and Microsoft since last month",
                    output_variable_name="news_developments",
                    is_output_node=False,
                ),
                ToolExecutionNode(
                    tool_name="collapse_lists",
                    args={"lists_of_lists": Variable(var_name="news_developments")},
                    description="Collapse the list of lists of news developments into a single list",
                    output_variable_name="collapsed_news_developments",
                    is_output_node=False,
                ),
                ToolExecutionNode(
                    tool_name="filter_texts_by_topic",
                    args={
                        "topic": "machine learning",
                        "texts": Variable(var_name="collapsed_news_developments"),
                    },
                    description="Filter news descriptions to only those related to machine learning",
                    output_variable_name="filtered_texts",
                    is_output_node=False,
                ),
                ToolExecutionNode(
                    tool_name="summarize_texts",
                    args={"texts": Variable(var_name="filtered_texts")},
                    description="Summarize the news descriptions into a single summary text",
                    output_variable_name="summary",
                    is_output_node=False,
                ),
                ToolExecutionNode(
                    tool_name="get_news_developments_about_companies",
                    args={
                        "stock_ids": [Variable(var_name="stock_ids", index=0)],
                        "start_date": Variable(var_name="start_date"),
                    },
                    description="Get news developments for only the first stock",
                    output_variable_name="unused",
                    is_output_node=False,
                ),
                ToolExecutionNode(
                    tool_name="get_name_of_single_stock",
                    args={
                        "stock_id": Variable(var_name="stock_ids", index=0),
                    },
                    description="Gets the name of the stock indicated by the stock_id",
                    output_variable_name="unused2",
                    is_output_node=False,
                ),
                ToolExecutionNode(
                    tool_name="prepare_output",
                    args={"object_to_output": [Variable(var_name="summary")], "title": "test"},
                    description="Output the result",
                    output_variable_name="result",
                    is_output_node=True,
                ),
            ]
        )
        plan_run_context = PlanRunContext.get_dummy()
        db = get_psql(skip_commit=True)
        agent = AgentMetadata(
            agent_id=plan_run_context.agent_id,
            user_id=plan_run_context.user_id,
            agent_name=DEFAULT_AGENT_NAME,
            created_at=get_now_utc(),
            last_updated=get_now_utc(),
            deleted=False,
        )
        db.insert_agent(agent)
        db.write_execution_plan(
            plan_id=plan_run_context.plan_id, agent_id=plan_run_context.agent_id, plan=plan
        )
        result = await run_execution_plan_local(
            plan,
            plan_run_context,
            do_chat=False,
        )
        self.assertIsNotNone(result)


class TestPlanConstructionValidation(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool_registry = get_test_registry()

    def test_literal_parsing(self):
        planner = Planner(agent_id="TEST")
        cases = [
            ("True", True),
            ("False", False),
            ("'Test one'", "Test one"),
            ('"Test"', "Test"),
            ("32520", 32520),
            ("3252.42", 3252.42),
            ("-32520", -32520),
            ("-3252.42", -3252.42),
        ]
        for in_str, expected in cases:
            self.assertEqual(
                expected, planner._try_parse_primitive_literal(in_str, expected_type=Any)
            )

    def test_list_parsing(self):
        planner = Planner(agent_id="TEST")
        variable_lookup = {"test1": int, "test2": str}
        cases = [
            ("[True]", [True], List[bool]),
            ("[True, False]", [True, False], List[bool]),
            ("[True,False]", [True, False], List[bool]),
            ("[True, 123,123.1]", [True, 123, 123.1], List[Union[bool, int, float]]),
            (
                "[True, 123,123.1, 'Hi']",
                [True, 123, 123.1, "Hi"],
                List[Union[bool, int, float, str]],
            ),
            ("[1, 2, test1]", [1, 2, Variable(var_name="test1")], List[int]),
            ("['1', '2', test2]", ["1", "2", Variable(var_name="test2")], List[str]),
        ]
        for in_str, expected, typ in cases:
            self.assertEqual(
                expected,
                planner._try_parse_list_literal(
                    in_str, expected_type=typ, variable_lookup=variable_lookup
                ),
            )

    def test_step_parse(self):
        planner = Planner(agent_id="TEST")
        example_input = """start_date = get_date_from_date_str(time_str="1 month ago")  # Convert "1 month ago" to a date to use as the start date for news search
stock_ids = stock_identifier_lookup_multi(stock_names=["Meta", "Apple", "Microsoft"])  # Look up stock identifiers for Meta, Apple, and Microsoft
news_developments = get_news_developments_about_companies(stock_ids=stock_ids, start_date=start_date)  # Get news developments in the last month for Meta, Apple, and Microsoft
collapsed_news_developments = collapse_lists(lists_of_lists=news_developments)  # Collapse the list of lists of news development IDs into a single list
filtered_news = filter_texts_by_topic(topic="machine learning", texts=collapsed_news_developments)  # Filter news descriptions to only those related to machine learning
summary = summarize_texts(texts=filtered_news)  # Summarize the machine learning related news into a single text"""  # noqa: E501
        steps = planner._parse_plan_str(example_input)

        expected_output = [
            ParsedStep(
                output_var="start_date",
                function="get_date_from_date_str",
                arguments={"time_str": '"1 month ago"'},
                description='Convert "1 month ago" to a date to use as the start date for news search',
            ),
            ParsedStep(
                output_var="stock_ids",
                function="stock_identifier_lookup_multi",
                arguments={"stock_names": '["Meta", "Apple", "Microsoft"]'},
                description="Look up stock identifiers for Meta, Apple, and Microsoft",
            ),
            ParsedStep(
                output_var="news_developments",
                function="get_news_developments_about_companies",
                arguments={"stock_ids": "stock_ids", "start_date": "start_date"},
                description="Get news developments in the last month for Meta, Apple, and Microsoft",
            ),
            ParsedStep(
                output_var="collapsed_news_developments",
                function="collapse_lists",
                arguments={"lists_of_lists": "news_developments"},
                description="Collapse the list of lists of news development IDs into a single list",
            ),
            ParsedStep(
                output_var="filtered_news",
                function="filter_texts_by_topic",
                arguments={"topic": '"machine learning"', "texts": "collapsed_news_developments"},
                description="Filter news descriptions to only those related to machine learning",
            ),
            ParsedStep(
                output_var="summary",
                function="summarize_texts",
                arguments={"texts": "filtered_news"},
                description="Summarize the machine learning related news into a single text",
            ),
        ]
        self.assertEqual(steps, expected_output)

    def test_validate_construct_plan(self):
        planner = Planner(agent_id="TEST")
        example_input = [
            ParsedStep(
                output_var="start_date",
                function="get_date_from_date_str",
                arguments={"time_str": '"1 month ago"'},
                description='Convert "1 month ago" to a date to use as the start date for news search',
            ),
            ParsedStep(
                output_var="stock_ids",
                function="stock_identifier_lookup_multi",
                arguments={"stock_names": '["Meta", "Apple,1", "Microsoft"]'},
                description="Convert company names to stock identifiers",
            ),
            ParsedStep(
                output_var="news_developments",
                function="get_news_developments_about_companies",
                arguments={"stock_ids": "stock_ids", "start_date": "start_date"},
                description="Get news developments for Meta, Apple, and Microsoft since last month",
            ),
            ParsedStep(
                output_var="collapsed_news_developments",
                function="collapse_lists",
                arguments={"lists_of_lists": "news_developments"},
                description="Collapse the list of lists of news developments into a single list",
            ),
            ParsedStep(
                output_var="filtered_news",
                function="filter_texts_by_topic",
                arguments={"topic": '"machine learning"', "texts": "collapsed_news_developments"},
                description="Filter news descriptions to only those related to machine learning",
            ),
            ParsedStep(
                output_var="summary",
                function="summarize_texts",
                arguments={"texts": "filtered_news"},
                description="Summarize the news descriptions into a single summary text",
            ),
            ParsedStep(
                output_var="unused",
                function="get_news_developments_about_companies",
                arguments={"stock_ids": "[stock_ids[0]]", "start_date": "start_date"},
                description="Get news developments for only the first stock",
            ),
            ParsedStep(
                output_var="unused2",
                function="get_name_of_single_stock",
                arguments={"stock_id": "stock_ids[0]"},
                description="Gets the name of the stock indicated by the stock_id",
            ),
            ParsedStep(
                output_var="result",
                function="prepare_output",
                arguments={"object_to_output": "summary", "title": '"test"'},
                description="Output the result",
            ),
        ]
        execution_plan = planner._validate_and_construct_plan(example_input)

        expected_output = [
            ToolExecutionNode(
                tool_name="get_date_from_date_str",
                args={"time_str": "1 month ago"},
                description='Convert "1 month ago" to a date to use as the start date for news search',
                output_variable_name="start_date",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="stock_identifier_lookup_multi",
                args={"stock_names": ["Meta", "Apple,1", "Microsoft"]},
                description="Convert company names to stock identifiers",
                output_variable_name="stock_ids",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="get_news_developments_about_companies",
                args={
                    "stock_ids": Variable(var_name="stock_ids"),
                    "start_date": Variable(var_name="start_date"),
                },
                description="Get news developments for Meta, Apple, and Microsoft since last month",
                output_variable_name="news_developments",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="collapse_lists",
                args={"lists_of_lists": Variable(var_name="news_developments")},
                description="Collapse the list of lists of news developments into a single list",
                output_variable_name="collapsed_news_developments",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="filter_texts_by_topic",
                args={
                    "topic": "machine learning",
                    "texts": Variable(var_name="collapsed_news_developments"),
                },
                description="Filter news descriptions to only those related to machine learning",
                output_variable_name="filtered_news",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="summarize_texts",
                args={"texts": Variable(var_name="filtered_news")},
                description="Summarize the news descriptions into a single summary text",
                output_variable_name="summary",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="get_news_developments_about_companies",
                args={
                    "stock_ids": [Variable(var_name="stock_ids", index=0)],
                    "start_date": Variable(var_name="start_date"),
                },
                description="Get news developments for only the first stock",
                output_variable_name="unused",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="get_name_of_single_stock",
                args={
                    "stock_id": Variable(var_name="stock_ids", index=0),
                },
                description="Gets the name of the stock indicated by the stock_id",
                output_variable_name="unused2",
                is_output_node=False,
            ),
            ToolExecutionNode(
                tool_name="prepare_output",
                args={"object_to_output": Variable(var_name="summary"), "title": "test"},
                description="Output the result",
                output_variable_name="result",
                is_output_node=True,
            ),
        ]

        for in_node, out_node in zip(execution_plan.nodes, expected_output):
            self.assertEqual(in_node.tool_name, out_node.tool_name)
            self.assertEqual(in_node.args, out_node.args)
            self.assertEqual(in_node.is_output_node, out_node.is_output_node)
            self.assertEqual(in_node.output_variable_name, out_node.output_variable_name)


class TestPlannerTypes(TestCase):
    def test_get_node_dependency_map(self):
        # define nodes
        test1 = ToolExecutionNode(
            tool_name="test_1",
            args={"name": "constant"},
            description="",
            output_variable_name="test_1_out",
        )
        test2 = ToolExecutionNode(
            tool_name="test_2",
            args={"name": "constant", "name2": Variable(var_name="test_1_out")},
            description="",
            output_variable_name="test_2_out",
        )
        test3 = ToolExecutionNode(
            tool_name="test_3",
            args={
                "name": "constant",
                "name2": Variable(var_name="test_1_out"),
                "name3": Variable(var_name="test_2_out"),
            },
            description="",
            output_variable_name="test_3_out",
        )
        test4 = ToolExecutionNode(
            tool_name="test_4",
            args={
                "name": Variable(var_name="test_1_out"),
            },
            description="",
            output_variable_name="test_4_out",
        )
        test5 = ToolExecutionNode(
            tool_name="test_5",
            args={
                "name": "constant",
                "name2": Variable(var_name="test_4_out"),
                "name3": Variable(var_name="test_3_out"),
            },
            description="",
            output_variable_name="test_5_out",
        )

        test_plan = ExecutionPlan(nodes=[test1, test2, test3, test4, test5])
        expected_tree = {
            test1: {test2, test3, test4},
            test2: {test3},
            test3: {test5},
            test4: {test5},
            test5: set(),
        }
        actual = test_plan.get_node_dependency_map()
        self.assertEqual(expected_tree, actual)

    def test_get_pruned_plan(self):
        # define nodes
        test1 = ToolExecutionNode(
            tool_name="test_1",
            args={"name": "constant"},
            description="",
            output_variable_name="test_1_out",
            tool_task_id="test_1",
        )
        test2 = ToolExecutionNode(
            tool_name="test_2",
            args={"name": "constant", "name2": Variable(var_name="test_1_out")},
            description="",
            output_variable_name="test_2_out",
        )
        test3 = ToolExecutionNode(
            tool_name="test_3",
            args={
                "name": "constant",
                "name2": Variable(var_name="test_1_out"),
                "name3": Variable(var_name="test_2_out"),
            },
            description="",
            output_variable_name="test_3_out",
        )
        test4 = ToolExecutionNode(
            tool_name="test_4",
            args={
                "name": Variable(var_name="test_1_out"),
            },
            description="",
            output_variable_name="test_4_out",
        )
        test5 = ToolExecutionNode(
            tool_name="test_5",
            args={
                "name": "constant",
                "name2": Variable(var_name="test_4_out"),
                "name3": Variable(var_name="test_3_out"),
            },
            description="",
            output_variable_name="test_5_out",
        )
        test_out_1 = ToolExecutionNode(
            tool_name="prepare_output",
            args={"name": Variable(var_name="test_2_out")},
            description="",
            is_output_node=True,
            tool_task_id="output1",
        )
        test_out_2 = ToolExecutionNode(
            tool_name="prepare_output",
            args={"name": Variable(var_name="test_5_out")},
            description="",
            is_output_node=True,
            tool_task_id="output2",
        )

        test_plan = ExecutionPlan(nodes=[test1, test2, test3, test4, test5, test_out_1, test_out_2])
        # First, remove output 1. Since another node depends on "test_2_out", it should have no effect
        expected_plan = ExecutionPlan(nodes=[test1, test2, test3, test4, test5, test_out_2])
        pruned = test_plan.get_pruned_plan(task_ids_to_remove={"output1"})
        with self.subTest(msg="No change expected"):
            self.assertEqual(pruned, expected_plan)

        # Now, remove output 2. Nodes should be removed in this case
        expected_plan = ExecutionPlan(nodes=[test1, test2, test_out_1])
        pruned = test_plan.get_pruned_plan(task_ids_to_remove={"output2"})
        with self.subTest(msg="Pruning expected"):
            self.assertEqual(pruned, expected_plan)

        # Now, remove both outputs. Plan should now be empty.
        expected_plan = ExecutionPlan(nodes=[])
        pruned = test_plan.get_pruned_plan(task_ids_to_remove={"output1", "output2"})
        with self.subTest(msg="All nodes removed"):
            self.assertEqual(pruned, expected_plan)

        # Now, remove nothing. Plan should be unchanged.
        expected_plan = ExecutionPlan(
            nodes=[test1, test2, test3, test4, test5, test_out_1, test_out_2]
        )
        pruned = test_plan.get_pruned_plan(task_ids_to_remove=set())
        with self.subTest(msg="No nodes removed"):
            self.assertEqual(pruned, expected_plan)

        with self.subTest(msg="Expect error removing non-output"):
            with self.assertRaises(RuntimeError):
                pruned = test_plan.get_pruned_plan(task_ids_to_remove={"test_1"})
