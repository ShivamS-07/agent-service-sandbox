import datetime
import unittest
import warnings
from typing import List, Type, Union
from unittest import IsolatedAsyncioTestCase

from agent_service.planner.planner import Planner
from agent_service.tools.io_types import ListofLists
from agent_service.tools.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc


def get_test_registry() -> Type[ToolRegistry]:
    class TestRegistry(ToolRegistry):
        pass

    class SummarizeTextInput(ToolArgs):
        texts: List[str]

    @tool(
        description="This function take a list of texts and use an LLM to summarize them into a single text",
        category=ToolCategory.LLM_ANALYSIS,
        tool_registry=TestRegistry,
    )
    async def summarize_texts(args: SummarizeTextInput, context: PlanRunContext) -> str:
        return ""

    class FilterByTopicRelevanceInput(ToolArgs):
        topic: str
        texts: List[str]

    @tool(
        description=(
            "This function takes any list of texts"
            " and uses machine learning to filter texts to only those relevant to the provided topic"
        ),
        category=ToolCategory.LLM_ANALYSIS,
        tool_registry=TestRegistry,
    )
    async def filter_by_topic_relevance(
        args: SummarizeTextInput, context: PlanRunContext
    ) -> List[str]:
        return []

    class GetNewsaboutCompaniesInput(ToolArgs):
        topic: str
        company_identifiers: List[int]
        start_date: datetime.date
        end_date: datetime.date = datetime.date.today()

    @tool(
        description=(
            "This function calls an interal API which provides all the news articles that were"
            "published between the start date and the end date that are relevant to the companies,"
            "the output is a list of list of articles (strings) in plain text,"
            "each list of articles corresponds to one of the companies"
        ),
        category=ToolCategory.DATA_RETRIEVAL_INT,
        tool_registry=TestRegistry,
    )
    async def get_news_about_companies(
        args: GetNewsaboutCompaniesInput, context: PlanRunContext
    ) -> ListofLists:
        return ListofLists(val=[[]])

    class CollapseListsInput(ToolArgs):
        lists_of_lists: ListofLists

    @tool(
        description="This function collapses a list of lists into a list",
        category=ToolCategory.LIST,
        tool_registry=TestRegistry,
    )
    async def collapse_lists(
        args: CollapseListsInput, context: PlanRunContext
    ) -> List[Union[str, int, float, bool]]:
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
        stock_str: str

    @tool(
        description="This function takes a string which refers to a stock, and converts it to an integer identifier",
        category=ToolCategory.STOCK,
        tool_registry=TestRegistry,
    )
    async def stock_identifier_lookup(
        args: StockIdentifierLookupInput, context: PlanRunContext
    ) -> int:
        return 0

    class StockIdentifierLookupMultiInput(ToolArgs):
        stock_strs: List[str]

    @tool(
        description=(
            "This function takes a list of strings each of which refers to a stock, "
            "and converts them to integer identifiers"
        ),
        category=ToolCategory.STOCK,
        tool_registry=TestRegistry,
    )
    async def stock_identifier_lookup_multi(
        args: StockIdentifierLookupMultiInput, context: PlanRunContext
    ) -> List[int]:
        return [0]

    return TestRegistry


class TestPlanner(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
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

    def setUp(self) -> None:
        self.tool_registry = get_test_registry()

    @unittest.skip("Takes too long to run")
    async def test_planner(self) -> None:
        input_text = (
            "Can you give me a single summary of news published in the last month "
            "about machine learning at Meta, Apple, and Microsoft?"
        )
        user_message = Message(content=input_text, is_user=True, timestamp=get_now_utc())
        chat_context = ChatContext(messages=[user_message])
        planner = Planner("", tool_registry=self.tool_registry)
        await planner.create_initial_plan(chat_context)
