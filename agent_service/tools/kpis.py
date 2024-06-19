import asyncio
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    Table,
    TableColumnMetadata,
)
from agent_service.io_types.text import EquivalentKPITexts, KPIText
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.types import ChatContext, Message, PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.kpi_extractor import KPIInstance, KPIMetadata, KPIRetriever
from agent_service.utils.postgres import get_psql
from agent_service.utils.prompt_utils import Prompt

kpi_retriever = KPIRetriever()
db = get_psql()

SPECIFIC = "specific"


@dataclass
class CompanyInformation:
    gbi_id: int
    company_name: str
    company_description: str
    kpi_lookup: Dict[int, KPIMetadata]
    kpi_str: str


DELIMITER = "-------------------------------------"
SEPERATOR = "_"

KPI_OVERLAPPING_KPIS_ACROSS_COMPANIES_SYS_PROMPT_OBJ = Prompt(
    name="KPI_OVERLAPPING_KPIS_ACROSS_COMPANIES_SYS_PROMPT",
    template=(
        "You are a financial analyst investigating the financial performance of companies have been "
        "tasked to identify key performance indicators (KPIs) that are equivalently comparable across "
        "a set of companies for a given topic, each company will be labeled by an index number. For each "
        "company you will be given a description of the company along with a set of KPIs that relate to "
        "the given topic. Each KPI shown to you will be presented on a single line with the following "
        "format '(kpi id) kpi name'.\n\n"
        "You must identify a KPI that appears across the set of companies that best reports on the "
        "topic in question however note that the specific wording of the KPI may differ across companies "
        "due to different product names or may be due to cases where some companies use 'sales' instead "
        "of 'revenue' in their KPI list. If you are given a set of companies where many companies contain 'Revenue' "
        "for a topic while others use 'Sales' or 'Net Sales' or vice versa, you may treat them as equivalent metrics "
        "and consider them to be equivalent providing they are reporting on the same topic. Most critically, when "
        "identifying equivalent KPIs you must find KPIs that report at the same level, for example, the "
        "KPIs 'Revenue - Apples' for one farming company and 'Revenue - Fruits' for another farming company "
        "are not equivalent as one KPI is broader than the other. You must try your best to find an equivalent "
        "KPI for each company shown to you unless you absolutely believe an equivalent KPI, as defined above, does "
        "not exist. Again, remember the specific wording of the equivalent KPI you're looking for may vary from "
        "company to company.\n\n"
        "Output your result as follows. The first line will contain a general name that accurately describes "
        "the equivalent KPI you have identified. Then each subsequent line will contain a specific KPI id from "
        "a company formatted as follows 'index{seperator}kpiID{seperator}justification' where index "
        "represents the index number associated with the company, kpiID is the kpi id associated with the kpi "
        "identified, and justification is a brief sentence that justifies why this is an equivalent kpi. "
        "You must output one line for each company shown to you. For companies without an equivalent kpi, "
        "output 0 as the kpiID and for the justification provide your reasoning for why you could not identify "
        "an equivalent kpi. Note that if you cannot find a KPI that is present across all companies you must aim "
        "to find a one that is present in the max number of companies. If you are unable to find any equivalent "
        "KPIs across any company simply output '0'."
    ),
)

KPI_OVERLAPPING_KPIS_ACROSS_COMPANIES_MAIN_PROMPT_OBJ = Prompt(
    name="KPI_OVERLAPPING_KPIS_ACROSS_COMPANIES_MAIN_PROMPT",
    template=(
        "Given the topic '{topic}', here are the following companies with their description and KPIs. "
        "Do not output any additional explanation or justification. "
        "Each company's descriptions and KPIs are delimited by '{delimiter}'\n\n"
        "{company_data}"
    ),
)


KPI_RELEVANCY_SYS_PROMPT_OBJ = Prompt(
    name="KPI_RELEVANCY_SYS_PROMPT",
    template=(
        "You are a financial analyst that has been tasked to identify relevant Key Performance Indicators "
        "(KPIs) for a given company. You will be told the specific KPIs in question, for example you may be "
        "asked to identify KPIs. surrounding a specific product or service, or you may be asked to identify "
        "KPIs that are the most impactful to the company's financial health/dominance. Unless explicitly told, "
        "you should avoid returning 'high-level' KPIs like 'Total Revenue' or 'EPS' and choose to return KPIs "
        "that are low-level meaning they refer to some specific product, service, etc."
    ),
)

KPI_SPECIFIC_KPI_SYS_PROMPT_OBJ = Prompt(
    name="KPI_SPECIFIC_KPI_SYS_PROMPT",
    template=(
        "You are a financial analyst that has been tasked to identify a single specific Key Performance Indicator "
        "(KPI) for a given company. You will be told the specific KPI in question, for example you may be "
        "asked to identify a KPI surrounding a specific product or service, or you may be asked to identify "
        "a KPI that is the most impactful to the company's financial health/dominance. Unless explicitly told, "
        "you should avoid returning 'high-level' KPIs like 'Total Revenue' or 'EPS' and choose to return a KPI "
        "that is low-level meaning it refers to some specific product, service, etc."
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
    "For the company {company_name}, identify relevant kpis around the following topic: '{query_topic}'. "
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

SPECIFIC_KPI_INSTRUCTION = (
    "For the company {company_name}, identify a single KPI that is most relevant to the topic '{query_topic}'. "
    "If there are multiple KPIs that are relevant, choose the one that is most impactful to the company. "
)
SPECIFIC_KPI_MAIN_PROMPT = Prompt(
    name="SPECIFIC_KPI_MAIN_PROMPT",
    template=(
        "{instructions}"
        "If you are unsure or there is not enough information given, defer to KPIs that measure "
        "revenue/sales and take the ones with the highest amount.\n\n"
        "To assist you, here is a brief description of {company_name}:\n"
        "{company_description}\n\n"
        "Below are the KPIs you are to select from, each KPI is presented on its own line with a  "
        "numerical identifier in brackets at the start, followed by the KPI name, followed by the  "
        "KPI value with its respective unit if applicable:\n"
        "{kpi_str}\n\n"
        "Return the single most relevant KPI by name and the numerical identifier associated with it."
        "Each line will contain a relevant KPI with the KPI number without the brackets"
        "followed by the KPI name separated by a comma."
        "Do not output any additional explanation or justification. Do not output more than 1 line / KPI."
    ),
)

CLASSIFY_SPECIFIC_GENERAL_MAIN_PROMPT = Prompt(
    name="CLASSIFY_SPECIFIC_GENERAL_MAIN_PROMPT",
    template=(
        "Given the topic '{topic}', determine if the topic is specific metric or if it is a more general "
        "product or service. A specific topic would be one that refers to a specific metric, like "
        "revenue or user growth, while a general topic would refer to a broader aspect of the company's assets "
        "or operations like a product. If the topic is specific, output 'Specific',"
        " if the topic is general, output 'General'"
    ),
)

CLASSIFY_SPECIFIC_GENERAL_SYS_PROMPT_OBJ = Prompt(
    name="CLASSIFY_SPECIFIC_GENERAL_SYS_PROMPT",
    template=(
        "You are a financial analyst that has been tasked to classify a topic as either a specific"
        "metric or a general product or service. You will be given a topic and must determine if the topic is "
        "specific or general. Output 'Specific' if the topic is specific, output 'General' if the topic is general."
    ),
)


def generate_columns_for_kpi(
    kpi_inst: KPIInstance,
    actual_col_name: str,
    estimate_col_name: str,
    surprise_col_name: str,
) -> List[TableColumnMetadata]:
    long_unit = kpi_inst.long_unit
    unit = kpi_inst.unit
    columns: List[TableColumnMetadata] = []
    if long_unit == "Amount":
        columns.append(
            TableColumnMetadata(label=actual_col_name, col_type=TableColumnType.CURRENCY)
        )
        columns.append(
            TableColumnMetadata(label=estimate_col_name, col_type=TableColumnType.CURRENCY)
        )
    elif long_unit == "Percent":
        columns.append(TableColumnMetadata(label=actual_col_name, col_type=TableColumnType.PERCENT))
        columns.append(
            TableColumnMetadata(label=estimate_col_name, col_type=TableColumnType.PERCENT)
        )
    else:
        columns.append(
            TableColumnMetadata(label=actual_col_name, col_type=TableColumnType.FLOAT, unit=unit)
        )
        columns.append(
            TableColumnMetadata(label=estimate_col_name, col_type=TableColumnType.FLOAT, unit=unit)
        )
    columns.append(TableColumnMetadata(label=surprise_col_name, col_type=TableColumnType.FLOAT))
    return columns


def convert_single_stock_data_to_table(
    stock_id: StockID, data: Dict[str, List[KPIInstance]]
) -> Table:
    data_dict: Dict[str, Any] = {}
    columns = [
        TableColumnMetadata(label="Quarter", col_type=TableColumnType.STRING),
        TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
    ]

    for kpi_name, kpi_history in data.items():
        actual_col = f"{kpi_name} Actual"
        estimate_col = f"{kpi_name} Estimate"
        surprise_col = f"{kpi_name} Surprise"
        new_columns = generate_columns_for_kpi(
            kpi_history[0], actual_col, estimate_col, surprise_col
        )
        columns.extend(new_columns)

        for kpi_inst in kpi_history:
            quarter = f"{kpi_inst.year} Q{kpi_inst.quarter}"

            if quarter not in data_dict:
                data_dict[quarter] = {
                    "Quarter": quarter,
                    STOCK_ID_COL_NAME_DEFAULT: stock_id,
                }

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
    for _, row in data_dict.items():
        df_data.append(row)

    df = pd.DataFrame(df_data)
    return Table.from_df_and_cols(data=df, columns=columns)


async def convert_multi_stock_data_to_table(
    kpi_name: str, data: Dict[StockID, List[KPIInstance]]
) -> Table:
    columns = []

    columns.append(TableColumnMetadata(label="Quarter", col_type=TableColumnType.STRING))
    columns.append(
        TableColumnMetadata(label=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK)
    )

    kpi_inst = list(data.values())[0][0]  # Take a random kpi_inst to pass in general kpi metadata
    actual_col_name = f"{kpi_name} (Actual)"
    estimate_col_name = f"{kpi_name} (Estimate)"
    surprise_col_name = f"{kpi_name} (Surpise)"
    kpi_column = generate_columns_for_kpi(
        kpi_inst=kpi_inst,
        actual_col_name=actual_col_name,
        estimate_col_name=estimate_col_name,
        surprise_col_name=surprise_col_name,
    )
    columns.extend(kpi_column)

    df_data = []
    for stock_id, kpi_history in data.items():
        for kpi_inst in kpi_history:
            quarter = f"{kpi_inst.year} Q{kpi_inst.quarter}"
            # Need to convert percentages into decimals to satisfy current handling of the Percentage figures
            actual = (
                kpi_inst.actual * 0.01
                if (kpi_inst.unit == "Percent" and kpi_inst.actual is not None)
                else kpi_inst.actual
            )
            estimate = kpi_inst.estimate * 0.01 if kpi_inst.unit == "Percent" else kpi_inst.estimate
            surprise = kpi_inst.surprise

            df_data.append(
                {
                    "Quarter": quarter,
                    STOCK_ID_COL_NAME_DEFAULT: stock_id,
                    actual_col_name: actual,
                    estimate_col_name: estimate,
                    surprise_col_name: surprise,
                }
            )

    df = pd.DataFrame(df_data)
    return Table.from_df_and_cols(data=df, columns=columns)


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


async def get_relevant_kpis_for_stock_id(
    stock_id: StockID, company_info: CompanyInformation, topic: str, llm: GPT
) -> List[KPIText]:
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
                KPIText(val=pid_metadata.name, id=pid_metadata.pid, stock_id=stock_id)
            )
    return topic_kpi_list


class GetKPIForStockGivenTopic(ToolArgs):
    stock_id: StockID
    topics: List[str]


@tool(
    description=(
        "This function will identify and return financial metrics reporting "
        "the key performance indicators (KPIs) that relate to a given topic for a given stock id. "
        "This function should only be invokved when a query makes mention of a singular stock, "
        "if multiple stocks are mentioned for the same topic or subject matter, use "
        "the get_relevant_kpis_for_multiple_stocks_given_topic function instead."
        "A stock_id must be provided to identify the stock for which important KPIs are "
        "fetched. A list of topic strings must also be provided to specify the topic of interest."
        "Even if there is only one topic it will be passed in as a list with one element. The "
        "topic string must be concise and make mention of some aspect or metric of a "
        "company's financials but must not mention the company name. "
        "The data returned will be a list of KPIText objects where each KPIText object "
        "contains one of the identified KPIs. Note that the KPIs returned are specific "
        "to the stock_id passed in, KPIText entries returned in the list must not be "
        "used interchangably or joined to other stock_id instances. Additionally, please "
        "note that this function does not provide any actual data for any given quarter for these KPIS."
        "You must use this function in conjunction with get_kpis_table_for_stock to"
        "retrieve the actual data for these KPIS."
    )
)
async def get_kpis_for_stock_given_topics(
    args: GetKPIForStockGivenTopic, context: PlanRunContext
) -> List[KPIText]:
    llm = GPT(context=None, model=GPT4_O)
    company_info = get_company_data_and_kpis(args.stock_id.gbi_id)
    stock_id = args.stock_id
    kpi_list = []
    for topic in args.topics:
        topic_kpi_list = await classify_specific_or_general_topic(
            stock_id, topic, company_info, llm
        )
        kpi_list.extend(topic_kpi_list)
    return kpi_list


class GetRelevantKPIsForStocksGivenTopic(ToolArgs):
    stock_ids: List[StockID]
    shared_metric: str


@tool(
    description=(
        "This function will identify and return financial metrics, key performance indicators "
        "(KPIs), across a set of companies the that best reports on a given metric. "
        "Use this function when a query refers to the same subject matter or metric over multiple stocks."
        "A list of stock ids must be provided via stock_ids to indicate the stocks "
        "to search for the given metric for. A shared_metric string must also be provided to specify the "
        "metric of interest. The shared_metric string must be consice and make mention of some aspect "
        "or metric of a company but must not mention any specific company's company name. "
        "The data returned will be a list of KPIText objects where each KPIText object "
        "contains an identified KPI from one company that best reports on the topic inputted. "
        "to the stock_id passed in, KPIText entries returned in the list must not be "
        "Note that this function may not identify a KPI for every company passed in through the stock_ids ."
        "list, however it will always at most return one KPI for each company, never more. Additionally, please "
        "note that this function does not provide any actual data for any given quarter for these KPIS."
    ),
    category=ToolCategory.KPI,
)
async def get_relevant_kpis_for_multiple_stocks_given_topic(
    args: GetRelevantKPIsForStocksGivenTopic, context: PlanRunContext
) -> EquivalentKPITexts:
    llm = GPT(context=None, model=GPT4_O)
    company_info_list: List[CompanyInformation] = []
    company_kpi_lists: List[List[KPIText]] = []
    overlapping_kpi_list: List[KPIText] = []
    company_lookup: Dict[str, CompanyInformation] = {}

    tasks = []
    gbi_id_stock_id_map = {}
    for stock_id in args.stock_ids:
        company_info = get_company_data_and_kpis(stock_id.gbi_id)
        company_info_list.append(company_info)
        tasks.append(
            get_relevant_kpis_for_stock_id(stock_id, company_info, args.shared_metric, llm)
        )
        gbi_id_stock_id_map[stock_id.gbi_id] = stock_id

    company_kpi_lists = await gather_with_concurrency(tasks, n=5)
    company_data = []
    for i, (company_info, company_kpi_list) in enumerate(zip(company_info_list, company_kpi_lists)):
        company_name = company_info.company_name
        company_description = company_info.company_description

        kpi_str_list = []
        for kpi in company_kpi_list:
            kpi_str_list.append(f"({kpi.id}) {kpi.val}")

        kpi_str = "\n".join(kpi_str_list)
        company_lookup[str(i + 1)] = company_info
        company_data.append(
            f"Company {i+1} - {company_name}\n"
            f"{company_name} Description: {company_description}\n\n"
            f"{company_name} KPIs:\n"
            f"{kpi_str}\n"
            f"{DELIMITER}"
        )
    result = await llm.do_chat_w_sys_prompt(
        main_prompt=KPI_OVERLAPPING_KPIS_ACROSS_COMPANIES_MAIN_PROMPT_OBJ.format(
            topic=args.shared_metric,
            company_data="\n".join(company_data),
            delimiter=DELIMITER,
        ),
        sys_prompt=KPI_OVERLAPPING_KPIS_ACROSS_COMPANIES_SYS_PROMPT_OBJ.format(
            seperator=SEPERATOR,
        ),
    )
    if result == "0":
        # GPT couldn't find anything
        return EquivalentKPITexts(val=[], general_kpi_name="")

    overlapping_kpi_gpt_resp = result.split("\n")
    # first line from gpt contains the generalized KPI name
    kpi_name = overlapping_kpi_gpt_resp[0]
    # retrieve the kpis
    for overlapping_kpi_data in overlapping_kpi_gpt_resp[1:]:
        # We force GPT to output a justification as well but we don't actually need it
        # ie. each line we ask it to output company_index,pid,justification
        company_index, pid, _ = overlapping_kpi_data.split(SEPERATOR)
        if pid == "0":
            # gpt could not find an equivalent kpi for this company
            continue
        gbi_id = company_lookup[company_index].gbi_id
        kpi_data = company_lookup[company_index].kpi_lookup.get(int(pid), None)
        if kpi_data is not None:
            overlapping_kpi_list.append(
                KPIText(
                    val=kpi_data.name, id=kpi_data.pid, stock_id=gbi_id_stock_id_map.get(gbi_id)
                )
            )

    return EquivalentKPITexts(val=overlapping_kpi_list, general_kpi_name=kpi_name)


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
        "contains one of the identified KPIs. Additionally, please "
        "note that this function does not provide any actual data for any given quarter for these KPIS."
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
                KPIText(val=pid_metadata.name, id=pid_metadata.pid, stock_id=args.stock_id)
            )
    return important_kpi_list


async def classify_specific_or_general_topic(
    stock_id: StockID, topic: str, company_info: CompanyInformation, llm: GPT
) -> List[KPIText]:

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=CLASSIFY_SPECIFIC_GENERAL_MAIN_PROMPT.format(topic=topic),
        sys_prompt=CLASSIFY_SPECIFIC_GENERAL_SYS_PROMPT_OBJ.format(),
    )
    if result.lower() == SPECIFIC:
        topic_kpi_list = await get_specific_kpi_data_for_stock_id(
            stock_id, company_info, [topic], llm
        )
    else:
        topic_kpi_list = await get_relevant_kpis_for_stock_id(stock_id, company_info, topic, llm)
    return topic_kpi_list


async def get_specific_kpi_data_for_stock_id(
    stock_id: StockID, company_info: CompanyInformation, topic: List[str], llm: GPT
) -> List[KPIText]:
    results = []
    instructions = SPECIFIC_KPI_INSTRUCTION.format(
        company_name=company_info.company_name, query_topic=topic
    )
    result = await llm.do_chat_w_sys_prompt(
        SPECIFIC_KPI_MAIN_PROMPT.format(
            instructions=instructions,
            company_name=company_info.company_name,
            company_description=company_info.company_description,
            kpi_str=company_info.kpi_str,
        ),
        KPI_RELEVANCY_SYS_PROMPT_OBJ.format(),
    )
    pid = int(result.split(",")[0])
    pid_metadata = company_info.kpi_lookup.get(pid, None)
    if pid_metadata:
        results.append(KPIText(val=pid_metadata.name, id=pid_metadata.pid, stock_id=stock_id))
    return results


def interpret_date_quarter_inputs(
    num_future_quarters: Optional[int] = None,
    num_prev_quarters: Optional[int] = None,
    anchor_date: Optional[datetime.date] = None,
    date_range: Optional[DateRange] = None,
) -> Tuple[int, int, datetime.datetime]:

    if date_range:
        anchor_date = date_range.end_date
        num_future_quarters = 0  # we will always do a look back instead of look forward
        num_days = (date_range.end_date - date_range.start_date).days
        days_in_quarter = 365.25 / 4
        # we always fetch 1 quarter for the anchor date so remove that one
        num_prev_days = num_days - days_in_quarter
        num_prev_days = max(0, num_prev_days)
        num_prev_quarters = round(num_prev_days / days_in_quarter)

    if num_future_quarters is None:
        num_future_quarters = 0

    # I think we should heavily consider setting this default to 1 or zero
    # (which ever would find the data for exactly the quarter containing the anchor_date)
    # and maybe change the name to quarter_containing_date
    if num_prev_quarters is None:
        num_prev_quarters = 7

    if anchor_date is None:
        anchor_datetime = get_now_utc()
    else:
        anchor_datetime = datetime.datetime.combine(anchor_date, datetime.datetime.min.time())

    return (num_future_quarters, num_prev_quarters, anchor_datetime)


class CompanyKPIsRequest(ToolArgs):
    stock_id: StockID
    kpis: List[KPIText]
    table_name: str
    num_future_quarters: Optional[int] = None
    num_prev_quarters: Optional[int] = None
    anchor_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function will fetch quarterly numerical data for a list of key performance indicators (KPIs) "
        "for a given company. A stock_id is required to indicate the company to grab the KPI data for. A list "
        "of kpis will be passed in via the kpis argument containing a list of KPIText objects to indicate "
        "the kpis to grab information for. The function must also take a table_name, this name should be brief "
        "and describe what the data represents (ie. 'Important KPIs for Apple' or 'Tesla KPIs Relating to Model X'). "
        "This function will always grab the data for the quarter associated with the anchor_date. "
        "The anchor_date should be a datetime.date object, not a date-like string, "
        "you should convert date strings into dates using the get_date_from_date_str function. "
        "If no anchor date is provided the function will assume anchor date is the present date "
        "or infer it from the date_range object if one was provided. Data from additional "
        "quarters can also be retrieved by specifying the num_prev_quarters, to indicate how many quarters prior to "
        "the year-quarter the anchor_date falls into. By default num_prev_quarters is set to 7. "
        "You can also specify the number of quarters after the anchor_date to grab data for by many consecutive "
        "quarters to grab data for by setting the num_future_quarters argument. By default num_future_quarters is "
        "set to 0. When a user requests KPI data for the last quarter, you may set num_future_quarters to 0 and "
        "num_prev_quarters to 1."
        " If a date_range is provided instead then num_future_quarters will be set to 0,"
        " anchor_date will be set to date_range.end_date,"
        " num_prev_quarters will be inferred from the width of the date_range."
    ),
    category=ToolCategory.KPI,
)
async def get_kpis_table_for_stock(args: CompanyKPIsRequest, context: PlanRunContext) -> Table:
    num_future_quarters, num_prev_quarters, anchor_date = interpret_date_quarter_inputs(
        args.num_future_quarters, args.num_prev_quarters, args.anchor_date, args.date_range
    )

    args.anchor_date = anchor_date
    args.num_future_quarters = num_future_quarters
    args.num_prev_quarters = num_prev_quarters

    kpi_metadata_dict = kpi_retriever.convert_kpi_text_to_metadata(
        gbi_id=args.stock_id.gbi_id, kpi_texts=args.kpis
    )
    kpi_list = list(kpi_metadata_dict.values())
    data = kpi_retriever.get_kpis_by_year_quarter_via_clickhouse(
        gbi_id=args.stock_id.gbi_id,
        kpis=kpi_list,
        starting_date=anchor_date,
        num_prev_quarters=args.num_prev_quarters,
        num_future_quarters=args.num_future_quarters,
    )

    topic_kpi_table = convert_single_stock_data_to_table(stock_id=args.stock_id, data=data)
    return topic_kpi_table


class KPIsRequest(ToolArgs):
    equivalent_kpis: EquivalentKPITexts
    table_name: str
    num_future_quarters: Optional[int] = None
    num_prev_quarters: Optional[int] = None
    anchor_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function will fetch quarterly numerical data for a list of key performance indicators (KPIs) "
        "across a set of companies. A list of kpis will be passed in via the kpis argument containing a list of "
        "KPIText objects to indicate the kpis to grab information for. The function must also take a table_name, "
        "this name should be brief and describe what the data represents (ie. 'Cloud Revenue' or "
        "'Automotive Sales'). This function will always grab the data for the quarter associated with the "
        "anchor_date. If no anchor date is provided the function will assume anchor date is the present date. "
        "Data from additional quarters can also be retrieved by specifying the num_prev_quarters, to indicate how "
        "many quarters prior to the year-quarter the anchor_date falls into. By default num_prev_quarters is "
        "set to 7. You can also specify the number of quarters after the anchor_date to grab data for by many "
        "consecutive quarters to grab data for by setting the num_future_quarters argument. By default "
        "num_future_quarters is set to 0. When a user requests KPI data for the last quarter, you may set "
        "num_future_quarters to 0 and num_prev_quarters to 1."
        " If a date_range is provided instead then num_future_quarters will be set to 0,"
        " anchor_date will be set to date_range.end_date,"
        " num_prev_quarters will be inferred from the width of the date_range."
    ),
    category=ToolCategory.KPI,
)
async def get_overlapping_kpis_table_for_stock(args: KPIsRequest, context: PlanRunContext) -> Table:
    num_future_quarters, num_prev_quarters, anchor_date = interpret_date_quarter_inputs(
        args.num_future_quarters, args.num_prev_quarters, args.anchor_date, args.date_range
    )

    args.anchor_date = anchor_date
    args.num_future_quarters = num_future_quarters
    args.num_prev_quarters = num_prev_quarters

    kpis: List[KPIText] = args.equivalent_kpis.val  # type: ignore

    company_kpi_data_lookup: Dict[StockID, List[KPIInstance]] = {}
    for kpi in kpis:
        if kpi.stock_id is None:
            continue

        kpi_metadata = kpi_retriever.convert_kpi_text_to_metadata(
            gbi_id=kpi.stock_id.gbi_id, kpi_texts=[kpi]
        ).get(kpi.id, None)

        if kpi_metadata is not None:
            data = kpi_retriever.get_kpis_by_year_quarter_via_clickhouse(
                gbi_id=kpi.stock_id.gbi_id,
                kpis=[kpi_metadata],
                starting_date=anchor_date,
                num_prev_quarters=args.num_prev_quarters,
                num_future_quarters=args.num_future_quarters,
            )
            kpi_data = data.get(kpi_metadata.name, None)
            if kpi_data:
                company_kpi_data_lookup[kpi.stock_id] = kpi_data

    topic_kpi_table = await convert_multi_stock_data_to_table(
        kpi_name=args.equivalent_kpis.general_kpi_name,
        data=company_kpi_data_lookup,
    )
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
    appl_stock_id = StockID(gbi_id=714, symbol="APPL", isin="")
    tr_stock_id = StockID(gbi_id=10753, symbol="TRI", isin="")
    specific_kpi = await get_specific_kpi_data_for_stock_id(  # type: ignore
        stock_id=tr_stock_id,
        company_info=get_company_data_and_kpis(tr_stock_id.gbi_id),
        topics=["thomson reuters legal profession revenue"],
        llm=GPT(context=None, model=GPT4_O),
    )
    specific_table: Table = await get_kpis_table_for_stock(  # type: ignore
        CompanyKPIsRequest(stock_id=tr_stock_id, table_name="TR Revenue", kpis=specific_kpi),
        context=plan_context,
    )
    df = specific_table.to_df()
    print(df.head())
    gen_kpi_list: List[KPIText] = await get_important_kpis_for_stock(  # type: ignore
        args=GetImportantKPIsForStock(stock_id=appl_stock_id), context=plan_context
    )
    gen_kpis_table: Table = await get_kpis_table_for_stock(  # type: ignore
        CompanyKPIsRequest(
            stock_id=appl_stock_id, table_name="General KPI Table", kpis=gen_kpi_list
        ),
        context=plan_context,
    )
    df = gen_kpis_table.to_df()
    print(df.head())
    for column in gen_kpis_table.columns:
        print(column.metadata.label)

    ford = StockID(gbi_id=4579, symbol="F", isin="")
    tesla = StockID(gbi_id=25508, symbol="TSLA", isin="")
    gm = StockID(gbi_id=25477, symbol="GM", isin="")
    rivian = StockID(gbi_id=520178, symbol="RIVN", isin="")

    stocks = [tesla, ford, gm, rivian]
    equivalent_kpis: EquivalentKPITexts = await get_relevant_kpis_for_multiple_stocks_given_topic(  # type: ignore
        GetRelevantKPIsForStocksGivenTopic(stock_ids=stocks, shared_metric="Automotive"),
        context=plan_context,
    )

    equivalent_kpis_table: Table = await get_overlapping_kpis_table_for_stock(  # type: ignore
        args=KPIsRequest(
            equivalent_kpis=equivalent_kpis,
            table_name="Automotive Revenue",
            num_future_quarters=0,
            num_prev_quarters=4,
        ),
        context=plan_context,
    )
    df = equivalent_kpis_table.to_df()
    print(df.head())
    for column in equivalent_kpis_table.columns:
        print(column.metadata.label)

    microsoft = StockID(gbi_id=6963, symbol="MSFT", isin="")
    amazon = StockID(gbi_id=149, symbol="AMZN", isin="")
    alphabet = StockID(gbi_id=10096, symbol="GOOG", isin="")

    stocks = [microsoft, amazon, alphabet]
    equivalent_kpis: EquivalentKPITexts = await get_relevant_kpis_for_multiple_stocks_given_topic(  # type: ignore
        GetRelevantKPIsForStocksGivenTopic(stock_ids=stocks, shared_metric="Cloud"),
        context=plan_context,
    )
    equivalent_kpis_table: Table = await get_overlapping_kpis_table_for_stock(  # type: ignore
        args=KPIsRequest(
            equivalent_kpis=equivalent_kpis,
            table_name="Cloud Computing",
            num_future_quarters=0,
            num_prev_quarters=4,
        ),
        context=plan_context,
    )
    df = equivalent_kpis_table.to_df()
    print(df.head())
    for column in equivalent_kpis_table.columns:
        print(column.metadata.label)


if __name__ == "__main__":
    asyncio.run(main())
