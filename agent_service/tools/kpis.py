import asyncio
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_types.misc import StockID
from agent_service.io_types.table import Table, TableColumn, TableColumnType
from agent_service.io_types.text import KPIText
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.kpi_extractor import KPIInstance, KPIMetadata, KPIRetriever
from agent_service.utils.postgres import get_psql
from agent_service.utils.prompt_utils import Prompt

kpi_retriever = KPIRetriever()
db = get_psql()


@dataclass
class CompanyInformation:
    stock_id: int
    company_name: str
    company_description: str
    kpi_lookup: Dict[int, KPIMetadata]
    kpi_str: str


KPI_RELEVANCY_SYS_PROMPT_OBJ = Prompt(
    name="KPI_RELEVANCY_SYS_PROMPT",
    template=(
        "You are a financial analyst that has been tasked to identify relevant Key Performance Indicators "
        "(KPIs) for a given company. You will be told the specific KPIs in question, for example you may be "
        "asked to identify KPIs. surrounding a specific product or service, or you may be asked to identify "
        "KPIs that are the most impactful to the company's financial health/dominance. Unless explicitly told, "
        "you should avoid returning 'high-level' KPIs like 'Total Revenue' or 'EPS' and choose to return KPIs "
        "that are low-level meaning they refer to some specific product, service, etc.)"
    ),
)

GENERAL_IMPORTANCE_INSTRUCTION_WITH_NUM_KPIS = (
    "For the company {company_name}, identify the {num_of_kpis} most "
    "important KPIs for the company. "
)

GENERAL_IMPORTANCE_INSTRUCTION = (
    "For the company {company_name}, identify the most important KPIs for the company. Limit yourself to a max "
    "of 10 KPIs. "
)

RELEVANCY_INSTRUCTION = (
    "For the company {company_name}, identify relevant kpis around the following topic: '{query_topic}'"
    "Limit yourself to at most 20 KPIs. You should not aim to get 20, stop when you feel that you have identified "
    "All relevant KPIs to the topic, however if you feel there are more than 20, return the 20 most impactful "
    "KPIs. "
)

KPI_RELEVANCY_MAIN_PROMPT = Prompt(
    name="KPI_RETRIEVAL_MAIN_PROMPT",
    template=(
        "{instructions}"
        "If you are unsure or there is not enough information given, defer to KPIs that measure "
        "revenue/sales and take the ones with the highest amount.\n\n"
        "To assist you, here is a brief description of {company_name}:\n"
        "{company_description}\n\n"
        "Below are the KPIs you are to select from, each KPI is presented on its own line with a numerical identifier "
        "in brackets at the start, followed by the KPI name, followed by the KPI value with its respective unit if "
        "applicable:\n"
        "{kpi_str}\n\n"
        "Return the relevant KPIs by name and the numerical identifier associated with it. Each line will contain a "
        "relevant KPI with the KPI number without the brackets followed by the KPI name separated by a comma. Do not "
        "output any additional explanation or justification. Do not output more than 10 lines."
    ),
)


def convert_data_to_table(title: str, data: Dict[str, List[KPIInstance]]) -> Table:
    data_dict: Dict[str, Any] = {}
    columns = [TableColumn(label="Quarter", col_type=TableColumnType.STRING)]
    for kpi_name, kpi_history in data.items():
        actual_col = f"{kpi_name} Actual"
        estimate_col = f"{kpi_name} Estimate"
        surprise_col = f"{kpi_name} Surprise"

        unit = kpi_history[0].unit
        if unit == "Amount":
            columns.append(TableColumn(label=actual_col, col_type=TableColumnType.CURRENCY))
            columns.append(TableColumn(label=estimate_col, col_type=TableColumnType.CURRENCY))
        elif unit == "Percent":
            columns.append(TableColumn(label=actual_col, col_type=TableColumnType.PERCENT))
            columns.append(TableColumn(label=estimate_col, col_type=TableColumnType.PERCENT))
        else:
            columns.append(
                TableColumn(
                    label=actual_col, col_type=TableColumnType.FLOAT, unit=kpi_history[0].unit
                )
            )
            columns.append(
                TableColumn(
                    label=estimate_col, col_type=TableColumnType.FLOAT, unit=kpi_history[0].unit
                )
            )
        columns.append(TableColumn(label=surprise_col, col_type=TableColumnType.FLOAT))

        for kpi_inst in kpi_history:
            quarter = f"Q{kpi_inst.quarter}-{kpi_inst.year}"

            if quarter not in data_dict:
                data_dict[quarter] = {"Quarter": quarter}

            # Need to convert percentages into decimals to satisfy current handling of the Percentage figures
            data_dict[quarter][actual_col] = (
                kpi_inst.actual * 0.01
                if (kpi_inst.unit == "Percent" and kpi_inst.actual is not None)
                else kpi_inst.actual
            )
            data_dict[quarter][estimate_col] = (
                kpi_inst.estimate * 0.01 if kpi_inst.unit == "Percent" else kpi_inst.estimate
            )
            data_dict[quarter][surprise_col] = kpi_inst.surprise

    # Convert the data dictionary to a list of rows
    df_data = []
    for quarter, row in data_dict.items():
        df_data.append(row)

    df = pd.DataFrame(df_data)
    return Table(title=title, data=df, columns=columns)


def get_company_data_and_kpis(gbi_id: int) -> CompanyInformation:
    company_name = db.get_sec_metadata_from_gbi([gbi_id])[gbi_id].company_name
    short_description = db.get_short_company_description(gbi_id)[0]

    # If we're unable to get the short description then pass in an empty string
    if short_description is None:
        short_description = ""

    kpis, kpi_data = kpi_retriever.get_all_company_kpis_current_year_quarter_via_clickhouse(
        gbi_id=gbi_id,
    )

    kpi_str_list = []
    for kpi in kpis:
        kpi_insts = kpi_data.get(kpi.pid, [])
        if len(kpi_insts):
            kpi_inst = kpi_insts[0]
            if kpi_inst.long_unit == "Amount":
                kpi_str_list.append(f"({kpi.pid}) {kpi.name}: ${kpi_inst.value:.2f}")
            elif kpi_inst.long_unit == "Number":
                kpi_str_list.append(f"({kpi.pid}) {kpi.name}: {kpi_inst.value:3f}")
            else:
                kpi_str_list.append(f"({kpi.pid}) {kpi.name}: {kpi_inst.value:.2f} {kpi_inst.unit}")

    kpi_lookup = {kpi.pid: kpi for kpi in kpis}
    kpi_str = "\n".join(kpi_str_list)
    return CompanyInformation(gbi_id, company_name, short_description, kpi_lookup, kpi_str)


async def get_relevant_kpis_for_gbi_id(
    gbi_id: int, company_info: CompanyInformation, topic: str
) -> List[KPIText]:
    llm = GPT(context=None, model=GPT4_O)
    instructions = RELEVANCY_INSTRUCTION.format(
        company_name=company_info.company_name, query_topic=topic
    )
    result = await llm.do_chat_w_sys_prompt(
        KPI_RELEVANCY_MAIN_PROMPT.format(
            instructions=instructions,
            company_name=company_info.company_name,
            company_description=company_info.company_description,
            kpi_str=company_info.kpi_str,
        ),
        KPI_RELEVANCY_SYS_PROMPT_OBJ.format(),
    )
    topic_kpi_list: List[KPIText] = []

    for line in result.split("\n"):
        pid = int(line.split(",")[0])
        # Check to make sure we have an actual pid that can map to data
        pid_metadata = company_info.kpi_lookup.get(pid, None)
        if pid_metadata:
            topic_kpi_list.append(
                KPIText(val=pid_metadata.name, id=pid_metadata.pid, stock_id=gbi_id)
            )
    return topic_kpi_list


class GetRelevantKPIsForStockGivenTopic(ToolArgs):
    stock_id: StockID
    topic: str


@tool(
    description=(
        "This function will identify and return financial metrics reporting "
        "the key performance indicators (KPIs) that relate to a given topic for a given stock id. "
        "A stock_id must be provided to identify the stock for which important KPIs are "
        "fetched. A topic string must also be provided to specify the topic of interest. The "
        "topic string must be consice and make mention of some aspect or metric of a "
        "company's financials but must not mention the company name. "
        "The data returned will be a list of KPIText objects where each KPIText object "
        "contains one of the identified KPIs. Note that the KPIs returned are specific "
        "to the stock_id passed in, KPIText entries returned in the list must not be "
        "used interchangably or joined to other stock_id instances."
    ),
    category=ToolCategory.KPI,
)
async def get_relevant_kpis_for_stock_given_topic(
    args: GetRelevantKPIsForStockGivenTopic, context: PlanRunContext
) -> List[KPIText]:
    company_info = get_company_data_and_kpis(args.stock_id.gbi_id)
    topic_kpi_list = await get_relevant_kpis_for_gbi_id(
        args.stock_id.gbi_id, company_info, args.topic
    )
    return topic_kpi_list


class GetImportantKPIsForStock(ToolArgs):
    stock_id: StockID
    num_of_kpis: Optional[int] = None


@tool(
    description=(
        "This function will identify and return financial metrics reporting "
        "the most important key performance indicators (KPIs) for a given stock id. "
        "A stock_id must be provided to identify the stock for which important KPIs are "
        "fetched. Optionally, an argument for the specific amount of kpis to be retrieved "
        "can be specified via the num_of_kpis argument. num_of_kpis does not need to be specified "
        "however. The default behavior will return the most important KPIs up to a limit of 10. "
        "The data returned will be a list of KPIText objects where each KPIText object "
        "contains one of the identified KPIs."
    ),
    category=ToolCategory.KPI,
)
async def get_important_kpis_for_stock(
    args: GetImportantKPIsForStock, context: PlanRunContext
) -> List[KPIText]:
    company_info = get_company_data_and_kpis(args.stock_id.gbi_id)
    llm = GPT(context=None, model=GPT4_O)

    if args.num_of_kpis is not None:
        instructions = GENERAL_IMPORTANCE_INSTRUCTION_WITH_NUM_KPIS.format(
            company_name=company_info.company_name, num_of_kpis=args.num_of_kpis
        )
        result = await llm.do_chat_w_sys_prompt(
            KPI_RELEVANCY_MAIN_PROMPT.format(
                instructions=instructions,
                company_name=company_info.company_name,
                company_description=company_info.company_description,
                kpi_str=company_info.kpi_str,
            ),
            KPI_RELEVANCY_SYS_PROMPT_OBJ.format(),
        )
    else:
        instructions = GENERAL_IMPORTANCE_INSTRUCTION.format(company_name=company_info.company_name)
        result = await llm.do_chat_w_sys_prompt(
            KPI_RELEVANCY_MAIN_PROMPT.format(
                instructions=instructions,
                company_name=company_info.company_name,
                company_description=company_info.company_description,
                kpi_str=company_info.kpi_str,
            ),
            KPI_RELEVANCY_SYS_PROMPT_OBJ.format(),
        )
    important_kpi_list: List[KPIText] = []

    for line in result.split("\n"):
        pid = int(line.split(",")[0])
        # Check to make sure we have an actual pid that can map to data
        pid_metadata = company_info.kpi_lookup.get(pid, None)
        if pid_metadata:
            important_kpi_list.append(
                KPIText(val=pid_metadata.name, id=pid_metadata.pid, stock_id=args.stock_id.gbi_id)
            )
    return important_kpi_list


class CompanyKPIsRequest(ToolArgs):
    stock_id: StockID
    kpis: List[KPIText]
    table_name: str
    num_future_quarters: int = 0
    num_prev_quarters: int = 7
    starting_date: Optional[datetime.datetime] = None


@tool(
    description=(
        "This function will fetch quarterly numerical data for a list of key performance indicators (KPIs) "
        "for a given company. A stock_id is required to indicate the company to grab the KPI data for. A list "
        "of kpis will be passed in via the kpis argument containing a list of KPIText objects to indicate "
        "the kpis to grab information for. The function must also take a table_name, this name should be brief "
        "and describe what the data represents (ie. 'Important KPIs for Apple' or 'Tesla KPIs Relating to Model X'). "
        "This function will always grab the data for the quarter associated with the starting_date. If no "
        "starting date is provided the function will assume starting date is the present date. Data from additional "
        "quarters can also be retrieved by specifying the num_prev_quarters, to indicate how many quarters prior to "
        "the year-quarter the starting_date falls into. By default num_prev_quarters is set to 7. "
        "You can also specify the number of quarters after the starting_date to grab data for by many consecutive "
        "quarters to grab data for by setting the num_future_quarters argument. By default num_future_quarters is "
        "set to 0."
    ),
    category=ToolCategory.KPI,
)
async def get_kpis_data_for_stock(args: CompanyKPIsRequest, context: PlanRunContext) -> Table:
    starting_date = get_now_utc() if args.starting_date is None else args.starting_date
    kpi_metadata_dict = kpi_retriever.convert_kpi_text_to_metadata(
        gbi_id=args.stock_id.gbi_id, kpi_texts=args.kpis
    )

    kpi_list = list(kpi_metadata_dict.values())
    data = kpi_retriever.get_kpis_by_year_quarter_via_clickhouse(
        gbi_id=args.stock_id.gbi_id,
        kpis=kpi_list,
        starting_date=starting_date,
        num_prev_quarters=args.num_prev_quarters,
        num_future_quarters=args.num_future_quarters,
    )
    topic_kpi_table = convert_data_to_table(title=args.table_name, data=data)
    return topic_kpi_table


async def main() -> None:
    # Useful testing data
    # apple_test = {"gbi_id": 714, "topic": "Apple TV"}
    # tesla_test = {"gbi_id": 25508, "topic": "Model X"}
    # dollarama_test = {"gbi_id": 23425, "topic": "Store Sales"}
    # roblox_test = {"gbi_id": 503877, "topic": "In Game Spending Per User"}

    input_text = "Hello :)"
    user_message = Message(message=input_text, is_user_message=True, message_time=get_now_utc())
    chat_context = ChatContext(messages=[user_message])
    plan_context = PlanRunContext(
        agent_id="123",
        plan_id="123",
        user_id="123",
        plan_run_id="123",
        chat=chat_context,
        run_tasks_without_prefect=True,
        skip_db_commit=True,
    )
    stock_id = StockID(gbi_id=714, symbol="APPL", isin="")

    gen_kpi_list: List[KPIText] = await get_important_kpis_for_stock(  # type: ignore
        args=GetImportantKPIsForStock(stock_id=stock_id), context=plan_context
    )
    topic_kpi_list: List[KPIText] = await get_relevant_kpis_for_stock_given_topic(  # type: ignore
        GetRelevantKPIsForStockGivenTopic(stock_id=stock_id, topic="Apple TV"),
        context=plan_context,
    )

    gen_kpis_table: Table = await get_kpis_data_for_stock(  # type: ignore
        CompanyKPIsRequest(stock_id=stock_id, table_name="General KPI Table", kpis=gen_kpi_list),
        context=plan_context,
    )
    print(gen_kpis_table.data.head())
    for column in gen_kpis_table.columns:
        print(column.label)
    print("-------------------------------------------------")

    topic_kpis_table: Table = await get_kpis_data_for_stock(  # type: ignore
        CompanyKPIsRequest(
            stock_id=stock_id, table_name="Topic KPI Table (Apple TV)", kpis=topic_kpi_list
        ),
        context=plan_context,
    )
    print(topic_kpis_table.data.head())
    for column in topic_kpis_table.columns:
        print(column.label)


if __name__ == "__main__":
    asyncio.run(main())
