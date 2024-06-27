# flake8: noqa
from agent_service.utils.prompt_utils import Prompt

OVERLAPPING_KPIS_ACROSS_COMPANIES_SYS_PROMPT = """You are a financial analyst investigating the financial performance of companies have been tasked to identify key performance indicators (KPIs) that are equivalently comparable across a set of companies for a given topic, each company will be labeled by an index number. For each company you will be given a description of the company along with a set of KPIs that relate to the given topic. Each KPI shown to you will be presented on a single line with the following format '(kpi id) kpi name'.

You must identify a KPI that appears across the set of companies that best reports on the topic in question however note that the specific wording of the KPI may differ across companies due to different product names or may be due to cases where some companies use 'sales' instead of 'revenue' in their KPI list. If you are given a set of companies where many companies contain 'Revenue' for a topic while others use 'Sales' or 'Net Sales' or vice versa, you may treat them as equivalent metrics and consider them to be equivalent providing they are reporting on the same topic. Most critically, when identifying equivalent KPIs you must find KPIs that report at the same level, for example, the KPIs 'Revenue - Apples' for one farming company and 'Revenue - Fruits' for another farming company are not equivalent as one KPI is broader than the other. You must try your best to find an equivalent KPI for each company shown to you unless you absolutely believe an equivalent KPI, as defined above, does not exist. Again, remember the specific wording of the equivalent KPI you're looking for may vary from company to company.

Output your result as follows. The first line will contain a general name that accurately describes the equivalent KPI you have identified. Then each subsequent line will contain a specific KPI id from a company formatted as follows 'index{seperator}kpiID{seperator}justification' where index represents the index number associated with the company, kpiID is the kpi id associated with the kpi identified, and justification is a brief sentence that justifies why this is an equivalent kpi. You must output one line for each company shown to you. For companies without an equivalent kpi, output 0 as the kpiID and for the justification provide your reasoning for why you could not identify an equivalent kpi. Note that if you cannot find a KPI that is present across all companies you must aim to find a one that is present in the max number of companies. If you are unable to find any equivalent KPIs across any company simply output '0'."""

OVERLAPPING_KPIS_ACROSS_COMPANIES_MAIN_PROMPT = """Given the topic '{topic}', here are the following companies with their description and KPIs. Do not output any additional explanation or justification. Each company's descriptions and KPIs are delimited by '{delimiter}'

{company_data}"""

KPI_RELEVANCY_SYS_PROMPT = """You are a financial analyst that has been tasked to identify relevant Key Performance Indicators (KPIs) for a given company. You will be told the specific KPIs in question, for example you may be asked to identify KPIs. surrounding a specific product or service, or you may be asked to identify KPIs that are the most impactful to the company's financial health/dominance. Unless explicitly told that the KPI you are looking for is a 'high-level' KPI, you should never returning 'high-level' KPIs like 'Total Revenue', 'Net Revenue', or 'EPS'. You should always aim to return KPIs that are low-level meaning they refer to some specific product, service, etc. That being said, if you are directly asked for these KPIs word for word. That is to say, if the input given to you is 'EPS' then of course you should return EPS. Similarly if you are explicitly asked for 'Total Revenue' then you should return a KPI that captures total revenue. However, you should not return 'total revenue' or 'net revenue' for things like 'total revenue from iPhone' or 'sales from computers'."""

GENERAL_IMPORTANCE_INSTRUCTION_WITH_NUM_KPIS = (
    "For the company {company_name}, identify the {num_of_kpis} most "
    "important KPIs for the company. "
)

GENERAL_IMPORTANCE_INSTRUCTION = (
    "For the company {company_name}, identify the most important KPIs for the company. Limit yourself to a max "
    "of 10 KPIs. "
)

RELEVANCY_INSTRUCTION = """For the company {company_name}, identify relevant kpis around the following topic: '{query_topic}'. Limit yourself to at most 20 KPIs. You should not aim to get 20, stop when you feel that you have identified All relevant KPIs to the topic, however if you feel there are more than 20, return the 20 most impactful KPIs. If you are unsure or there is not enough information given, defer to KPIs that measure revenue/sales and take the ones with the highest amount."""

SPECIFIC_KPI_INSTRUCTION = """For the company {company_name}, identify a single KPI that is most relevant to the metric '{query_topic}'. If there are multiple KPIs that are relevant, choose the one that is most impactful to the company. Never output more than one KPI in your output. If you cannot find a KPI that directly reports on the topic given to you return an empty string without any explanation or justification. You are not looking for a direct word-for-word match onto '{query_topic}'. You are simply looking at whether or not any of the KPIs shown to you report on this given metric."""

KPI_RELEVANCY_MAIN_PROMPT = """{instructions}

To assist you, here is a brief description of {company_name}:
{company_description}

Below are the KPIs you are to select from, each KPI is presented on its own line with a numerical identifier in brackets at the start, followed by the KPI name, followed by the KPI value with its respective unit if applicable:
{kpi_str}

Return the relevant KPIs by name and the numerical identifier associated with it. Each line will contain a relevant KPI with the KPI number without the brackets followed by the KPI name separated by a comma which must be followed by a justification as to why the selected KPI directly reports on the topic given also separated by a comma. Your output should look like:
123123, KPI NAME, JUSTIFICATION

Where 123123 is the number associated with the KPI, KPI NAME is the name of the KPI, and JUSTIFICATION is the justification. Do not use commas anywhere else other than to separate the three outputs in your response."""

CLASSIFY_SPECIFIC_GENERAL_MAIN_PROMPT = """Given the topic '{topic}', determine if the topic is specific metric or if it is a more general product or service. A specific topic would be one that refers to a specific metric, like revenue or user growth, while a general topic would refer to a broader aspect of the company's assets or operations like a product. If the topic is specific, output 'Specific', if the topic is general, output 'General'"""

CLASSIFY_SPECIFIC_GENERAL_SYS_PROMPT = """You are a financial analyst that has been tasked to classify a topic as either a specific metric or a general product or service. You will be given a topic and must determine if the topic is specific or general. Any topic that involves a measurement such as 'revenue', 'sales', 'cost', 'subscribers', and so on must be considered a metric and thus should be classified as specific. If the topic in question simply names what appears to be a product or area of a company's business such as 'Advertisment' or 'Disney+' then you classify these as general.  Output 'Specific' if the topic is specific, output 'General' if the topic is general."""


OVERLAPPING_KPIS_ACROSS_COMPANIES_SYS_PROMPT_OBJ = Prompt(
    name="KPI_OVERLAPPING_KPIS_ACROSS_COMPANIES_SYS_PROMPT",
    template=OVERLAPPING_KPIS_ACROSS_COMPANIES_SYS_PROMPT,
)

OVERLAPPING_KPIS_ACROSS_COMPANIES_MAIN_PROMPT_OBJ = Prompt(
    name="KPI_OVERLAPPING_KPIS_ACROSS_COMPANIES_MAIN_PROMPT",
    template=OVERLAPPING_KPIS_ACROSS_COMPANIES_MAIN_PROMPT,
)

KPI_RELEVANCY_SYS_PROMPT_OBJ = Prompt(
    name="KPI_RELEVANCY_SYS_PROMPT",
    template=KPI_RELEVANCY_SYS_PROMPT,
)

KPI_RELEVANCY_MAIN_PROMPT_OBJ = Prompt(
    name="KPI_RETRIEVAL_MAIN_PROMPT",
    template=KPI_RELEVANCY_MAIN_PROMPT,
)

SPECIFIC_KPI_MAIN_PROMPT_OBJ = Prompt(
    name="SPECIFIC_KPI_MAIN_PROMPT",
    template=KPI_RELEVANCY_MAIN_PROMPT,
)

CLASSIFY_SPECIFIC_GENERAL_MAIN_PROMPT_OBJ = Prompt(
    name="CLASSIFY_SPECIFIC_GENERAL_MAIN_PROMPT",
    template=CLASSIFY_SPECIFIC_GENERAL_MAIN_PROMPT,
)

CLASSIFY_SPECIFIC_GENERAL_SYS_PROMPT_OBJ = Prompt(
    name="CLASSIFY_SPECIFIC_GENERAL_SYS_PROMPT",
    template=CLASSIFY_SPECIFIC_GENERAL_SYS_PROMPT,
)

###### TOOL DESCRIPTIONS
IMPORTANT_KPIS_FOR_STOCK_DESC = """This function will identify and return financial metrics reporting key performance indicators (KPIs) for a given stock id when a user has not specified or provided any details as to the KPIs they are interested in or have stated want to see the most important KPIs of a specific company. If the user has specified an area or areas of interest within the company you must use the function get_kpis_for_stock_given_topics instead. A stock_id must be provided to identify the stock for which important KPIs are fetched. Optionally, an argument for the specific amount of kpis to be retrieved can be specified via the num_of_kpis argument. num_of_kpis does not need to be specified however. The default behavior will return the most important KPIs up to a limit of 10. The data returned will be a list of KPIText objects where each KPIText object contains one of the identified KPIs. Additionally, please note that this function does not provide any actual data for any given quarter for these KPIS, the Text output is simply the name of the KPI, e.g. "Cloud Revenue" and is useless for direct quantitative analysis.."""

GET_KPI_TABLE_FOR_STOCK_DESC = """This function will fetch quarterly numerical data for a list of key performance indicators (KPIs) for a given company. A stock_id is required to indicate the company to grab the KPI data for. A list of kpis will be passed in via the kpis argument containing a list of KPIText objects to indicate the kpis to grab information for. The function must also take a table_name, this name should be brief and describe what the data represents (ie. 'Important KPIs for Apple' or 'Tesla KPIs Relating to Model X'). This function will always grab the data for the quarter associated with the anchor_date. The anchor_date should be a datetime.date object, not a date-like string, you should convert date strings into dates using the get_date_from_date_str function. If no anchor date is provided the function will assume anchor date is the present date or infer it from the date_range object if one was provided. Data from additional quarters can also be retrieved by specifying the num_prev_quarters, to indicate how many quarters prior to the year-quarter the anchor_date falls into. By default num_prev_quarters is set to 7. You can also specify the number of quarters after the anchor_date to grab data for by many consecutive quarters to grab data for by setting the num_future_quarters argument. By default num_future_quarters is set to 0. When a user requests KPI data for the last quarter, you may set num_future_quarters to 0 and num_prev_quarters to 1. If a date_range is provided instead then num_future_quarters will be set to 0, anchor_date will be set to date_range.end_date, num_prev_quarters will be inferred from the width of the date_range. Note this is raw KPI data, often in dollars or other currency. If the user is asking for percentages or something similar you must transform this table! Lastly, if only one quarter of date is desired to be used to generate pie chart data set simple_output to True which will output a simplified table for a single quarters worth of data for the anchor_date."""

GET_OVERLAPPING_KPIS_TABLE_FOR_STOCK_DESC = """This function will fetch quarterly numerical data for a list of key performance indicators (KPIs) across a set of companies. A list of kpis will be passed in via the kpis argument containing a list of KPIText objects to indicate the kpis to grab information for. The function must also take a table_name, this name should be brief and describe what the data represents (ie. 'Cloud Revenue' or 'Automotive Sales'). This function will always grab the data for the quarter associated with the anchor_date. If no anchor date is provided the function will assume anchor date is the present date."Data from additional quarters can also be retrieved by specifying the num_prev_quarters, to indicate how many quarters prior to the year-quarter the anchor_date falls into. By default num_prev_quarters is set to 7. You can also specify the number of quarters after the anchor_date to grab data for by many"consecutive quarters to grab data for by setting the num_future_quarters argument. By default num_future_quarters is set to 0. When a user requests KPI data for the last quarter, you may set num_future_quarters to 0 and num_prev_quarters to 1. If a date_range is provided instead then num_future_quarters will be set to 0, anchor_date will be set to date_range.end_date, num_prev_quarters will be inferred from the width of the date_range. Lastly, if only one quarter of date is desired to be used to generate pie chart data set simple_output to True which will output a simplified table for a single quarters worth of data for the anchor_date."""

GET_KPIS_FOR_STOCK_GIVEN_TOPICS_DESC = """This function will identify and return financial metrics reporting the key performance indicators (KPIs) that relate to a given topic for a given stock id. This function should only be invokved when a query makes mention of a singular stock, if multiple stocks are mentioned for the same topic or subject matter, use the get_relevant_kpis_for_multiple_stocks_given_topic function instead. A stock_id must be provided to identify the stock for which important KPIs are fetched. A list of topic strings must also be provided to specify the topic of interest. Even if there is only one topic it will be passed in as a list with one element. The topic string must be concise and make mention of some aspect or metric of a company's financials but must not mention the company name. The data returned will be a list of KPIText objects where each KPIText object contains one of the identified KPIs. Note that the KPIs returned are specific to the stock_id passed in, KPIText entries returned in the list must not be used interchangably or joined to other stock_id instances. Additionally, please note that this function does not provide any actual data for any given quarter for these KPIS, the Text output is simply the name of the KPI, e.g. "Could Revenue" and is useless for direct quantitative analysis of any kind. You must use this function in conjunction with get_kpis_table_for_stock to retrieve the actual data for these KPIS."""

GET_RELEVANT_KPIS_FOR_MULTIPLE_STOCKS_GIVEN_TOPIC_DESC = """This function will identify and return financial metrics, key performance indicators (KPIs), across a set of companies the that best reports on a given metric. Use this function when a query refers to the same subject matter or metric over multiple stocks. A list of stock ids must be provided via stock_ids to indicate the stocks to search for the given metric for. A shared_metric string must also be provided to specify the metric of interest. The shared_metric string must be conscise and make mention of some aspect or metric of a company but must not mention any specific company's company name. The data returned will be a list of KPIText objects where each KPIText object contains an identified KPI from one company that best reports on the topic inputted. Note that this function may not identify a KPI for every company passed in through the stock_ids list, however it will always at most return one KPI for each company, never more. Additionally, please note that this function does not provide any actual data for any given quarter for these KPIS, the Text output is simply the name of the KPI, e.g. "Cloud Revenue", if you need to do quantative analysis you must call a tool to get a table."""
