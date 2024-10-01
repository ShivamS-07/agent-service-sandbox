# flake8: noqa

from agent_service.tools.LLM_analysis.prompts import (
    CITATION_PROMPT,
    CITATION_REMINDER,
    SIMPLE_PROFILE_DEFINITION,
)
from agent_service.utils.prompt_utils import Prompt

PROFILE_EXPOSURE_TEXT_EVALUATER_MAIN_PROMPT_STR = """The profile you have been given is as follows: "{profile}". Evaluate if there is some credible link between the company, {company_name}, and the profile given. If the profile includes a specific relationship with a particular company (ie. buyer/supplier relationships) as a condition, there must be explicit evidence found within the relevant documents that the company you're evaluating has that relationship with the specified company for there to be a potential credible link, if you cannot find an explicit mention of that relationship you must conclude no credible link exists. This is extremely important. If you believe a credible link exists output 1, if a connection does not exist, output 2. Follow this output with a justification for your de Below are all of the relevant documents:
{company_texts}"""

PROFILE_EXPOSURE_TEXT_EVALUATER_SYS_PROMPT_STR = "You are a financial analyst looking at a company to see if there is any credible links between a company and a description of a company you have been given, referred to as a profile. If the profile includes a description of a specific relationship with a company (ie. buyer/supplier relationships) there must be explicit evidence that relationship exists within the relevant documents shown to you, if there is no explicit mention of that relationship you must conclude no credible link exists. You will be provided with documents associated with the company, this may include things like the company's financial documents, earning transcripts or summaries of their earnings, relevent news published about the company, and more. You must look through these texts and decide on whether anything mentioned in these documents indicates the company may be a fit for the profile given to you. It is crucial to note that even if a company just barely mentions a small handful of things that fit to the type of company described in the profile you must conclude that a connection exists. If there is nothing in any of the documents that describe aspects of the company in a way that aligns with the profile given then you must conclude no connection exists."

PROFILE_EXPOSURE_TEXT_SUMMARIZER_MAIN_PROMPT_STR = (
    """You are a financial analyst highly skilled at analyzing documents for insights about a company. Specifically, you are interested in companies that seem to fit a specific description of a type of company, referred to as a profile. In this particular case, the profile in question is "{profile}". You have been given a company that may have components of its business that matches the subject specified. Below are a group of documents from the company, extract any relevant information that capture the exposure and focus to the pertaining to the profile given to you. The group of text documents is shown below and is delimited by "------":{documents}\n------\n"""
    + CITATION_REMINDER
)

PROFILE_EXPOSURE_TEXT_SUMMARIZER_SYS_PROMPT_STR = (
    """You are a financial analyst highly skilled at analyzing documents for insights about companies. You are interested in companies exposed to a specific subject which will be described to you through a profile. You will be given a group of documents which talk about a particular company, and need to identify and summarize any relevant information that provides insights as to the extent the company is exposed, focused, or committed to the description given to you as the profile. It is critical that you do not write anything overly positive or aim to positively market the company you have been given, you are not writing an advertisement for the company in relation to the given profile. You must be objective in reporting the facts that have been presented in the text documents you have been given, ignoring the more subjective things that may be included in these documents, especially documents that are derived by or are outright published by the company itself that may self-proclaim their technology as best in class or add overly positive spins to everything within their operations. In your summary you must also include insights into which aspects of the company align with the description described in the profile, as well as how focused or dependent the company are to those things and how it compares to their focus and dependency on the rest of their operations."""
    + CITATION_PROMPT
)

TWO_COMP_PROFILE_COMPARISON_MAIN_PROMPT_STR = """Determine which company is a better fit to the given company description of "{profile}".
The first company shown to you is {company1_name}, here are some relevant points that relate to the company description given, delimited by "------":
{company1_summary}
------
The second company is {company2_name}, here are some relevant points that relate to the company description given:
{company2_summary}
------
Output 1 if you believe that {company1_name} is the better fit, output 2 if you believe that {company2_name} is the the better fit. After this output, you must add a pipe delimiter and provide a brief but compelling justification for why you have made this decision. 

For example your output should take the format of "1|the first company has a...". You must take absolute care and diligence to ensure you have picked the best option.
"""

TWO_COMP_PROFILE_COMPARISON_SYS_PROMPT_STR = """You are a highly skilled financial analyst performing analysis on companies with a specific company description. Specifically, you will be comparing companies and identifying which one is a stronger fit to a given profile. When assessing fit, you must obviously assess a company's outright match to the profile but it is extremely critical that you must also assess the company's dependence, commitment, exposure, and focus on the profile, some profiles may be related to a companies operations, others will be focused on income streams, and other will target something else entirely. If the profile is focused on income stream then while it is important to determine which company is a stronger fit you should also take into account which company seems to depend on the income stream more. Similarly, if the profile focuses on a specific technology then in addition to how will the company's operations fit to the profile, the companies focus/commitment/dependence on the technology must also be taken into account. You will be shown two companies, each company will contain a summary highlighting their fit to the given profile. The company you pick should be the one that is a stronger match to the profile."""

TIEBREAKER_MAIN_PROMPT_STR = """Rank the following companies by order of strongest fit for the company profile, '{profile}'. Each company is denoted with the company name and an associated number followed by a summary of the various operations within the company relating to the given company description followed by a delimiter, "------". Output your ranking by listing out the company numbers in descending order, outputting a descending list starting with the company that is the best fit and ending with the company that is the worst fit, separated by a comma. For example, if given 5 companies labeled 1 to 5, you might output: 3,2,4,5,1

Do not output any additional explanation or justification, though you must be absolutely confident in your final answer and have thoroughly thought it through.

The companies are shown below, the order they are shown bear no significance to their potential fit to the company description and should not bias you in any way.
{companies_str}"""

TIEBREAKER_SYS_PROMPT_STR = """You are a highly skilled financial analyst performing research on a specific type of company, descriped through a "company profile". Specifically, you will be comparing companies and identifying and ranking these companies by strongest match to a given profile. A strong match to a profile changes slightly based on the nature of the profile. Some profiles may be related to a companies operations, others will be focused on income streams, and other will target something else entirely. If the profile is focused on income stream then while it is important to determine which company is a stronger fit you should also take into account which company seems to depend on the income stream more. Similarly, if the profile focuses on a specific technology then in addition to how will the company's operations fit to the profile, the companies focus/commitment/dependence on the technology must also be taken into account. You will be shown a list of companies, each company will contain a summary highlighting the parts of their operation that relate to the profile. Your job is to rank these companies by order of decreasing fit, starting with the company that is the best fit and ending with the company that is the worst fit."""


PROFILE_EXPOSURE_TEXT_SUMMARIZER_MAIN_PROMPT = Prompt(
    name="RELEVANT_TEXT_SUMMARIZER_MAIN_PROMPT",
    template=PROFILE_EXPOSURE_TEXT_SUMMARIZER_MAIN_PROMPT_STR,
)
PROFILE_EXPOSURE_TEXT_SUMMARIZER_SYS_PROMPT = Prompt(
    name="RELEVANT_TEXT_SUMMARIZER_SYS_PROMPT",
    template=PROFILE_EXPOSURE_TEXT_SUMMARIZER_SYS_PROMPT_STR,
)

PROFILE_EXPOSURE_TEXT_EVALUATER_MAIN_PROMPT = Prompt(
    name="RELEVANT_TEXT_SUMMARIZER_MAIN_PROMPT",
    template=PROFILE_EXPOSURE_TEXT_EVALUATER_MAIN_PROMPT_STR,
)
PROFILE_EXPOSURE_TEXT_EVALUATER_SYS_PROMPT = Prompt(
    name="RELEVANT_TEXT_SUMMARIZER_SYS_PROMPT",
    template=PROFILE_EXPOSURE_TEXT_EVALUATER_SYS_PROMPT_STR,
)

TWO_COMP_PROFILE_COMPARISON_MAIN_PROMPT = Prompt(
    name="RELEVANCY_TWO_COMP_COMPARISON_MAIN_PROMPT",
    template=TWO_COMP_PROFILE_COMPARISON_MAIN_PROMPT_STR,
)
TWO_COMP_PROFILE_COMPARISON_SYS_PROMPT = Prompt(
    name="RELEVANCY_TWO_COMP_COMPARISON_SYS_PROMPT",
    template=TWO_COMP_PROFILE_COMPARISON_SYS_PROMPT_STR,
)

TIEBREAKER_MAIN_PROMPT = Prompt(
    name="RELEVANCY_TIEBREAKER_MAIN_PROMPT",
    template=TIEBREAKER_MAIN_PROMPT_STR,
)
TIEBREAKER_SYS_PROMPT = Prompt(
    name="RELEVANCY_TIEBREAKER_SYS_PROMPT",
    template=TIEBREAKER_SYS_PROMPT_STR,
)

# Tool Descriptions
RANK_STOCKS_BY_PROFILE_DESCRIPTION = f"""You must invoke this function whenever a user expresses a desire to rank, score, or see what the top and/or bottom N companies are, with respect to some description of a specific type of company. If the user has expressed a desire to do this, then do not apply any filtering beyond possibly a sector filter via the sector_filter tool, and do not use any recommendation tool either such as get_stock_recommendations however under no circumstances are you allowed to filter by any profiles. This function must never be used alongside filter_stocks_by_profile_match nor is it ever allowed to take the outputs from filter_stocks_by_profile_match as its input. This function takes a list of stocks via the 'stocks' argument along with a list of texts associated with the input stocks via the 'stock_text' argument. Neither of these lists can be empty. The texts passed to this tool must be the output of a text retrieval tool such as `get_all_text_data_for_stocks` called with the same stocks in the `stocks` list as the argument. The purpose of this function is to rank and score a list of stocks based on the associated texts for each stock according to how strongly they match a the given company description. If a user has requested to rank a list of stocks by some common financial statistic like market cap, P/E ratio, returns, etc. must use the `transform_table` tool. This function must only be used when a user requests a ranking or requests of scoring for a list of stocks or get some top/bottom X best stocks, if the user has requested to filter a list of stocks you must use the `filter_stocks_by_profile_match` tool instead. The company description is passed in as a string through the 'profile' argument, {SIMPLE_PROFILE_DEFINITION} The output of this function returns the list of stocks sorted in descending order by the score they have been assigned, which is calculated based on how strongly each company fit the profile given. If the top N stocks are desired, a 'top_n' argument can be passed in, upon which this function will only return the 'top_n' stocks, also in descending order of relevancy/exposure. Only use this function when you want to rank stocks by some specific profile, do not use this function for any other purpose. Likewise, if the bottom M stocks are desired, a 'bottom_m' argument can be passed in, upon which this function will only return the 'bottom_m' stocks. If a user wants some top N and bottom M stocks to be returned you must specify both arguments when invoking this tool and the tool will return the top_n and bottom_m stocks. If this behavior is not desired by the user, do not pass in any value to either of these arugments and the full list of ranked stocks will be returned. It is extremely important to pass in a high quality string into the 'profile' argument as described earlier as this is what the stocks will be evaluated against. The more specific and detailed it is, the better the results will be.

If the user does not mention a specific kinds of texts to use for this filter (e.g. news, SEC filing, earnings call, custom document, etc.), just call the `get_default_text_data_for_stocks` tool before this one to get a list of all texts for the stocks. However, if the client specifically mentions a kind of text or kinds of texts to use in the filtering, then the list of texts passed in to this tool must include exactly those kinds of texts and no others. For example, if the client says `Give me stocks that mention buybacks in their earnings calls or sec filings`, then the `texts` argument to this function must ONLY include earnings calls summaries and sec filings, you must NOT pass all text data. I repeat: when the user does NOT specify the kind of documents that should be used, you must default to getting all documents for relevant stocks when you use this tool, but when they do, you must pass only the kinds of documents that the client wants to this tool. Never just use company descriptions unless that is what the client asks for! You should avoid calling this tool with huge numbers of stocks/texts (more than 1000). If a client has asked about fit to a given profile for all stocks within some large ETF or stock universe, you must filter the stocks down to the applicable stocks by using the sector_filter tool first, using an appropriate sector that well encapsulates the profile given to you! However you must not filter using the filter_stocks_by_profile_match as this will result in catestrophic failure."""


# Rubric Generation
# flake8: noqa

from agent_service.utils.prompt_utils import Prompt

PROFILE_RUBRIC_GENERATION_SYS_PROMPT_STR = """You are a highly skilled financial analyst. You have been tasked with designing a scoring rubric for how well a particular company matches with a given 'profile'. A 'profile' can be a couple of different things but fundamentally it is a multi-paragraph description or summarization of a particular industry, sector, or field. Your scoring rubric will be broken down into five distinct levels, 0 to 5. Where level 0 is any company that has absolutely zero relation to the 'profile' in question, level 1 is given to any company that barely matches the information given in the profile at all and a level 5 describes a company that is nearly a perfect match for the kind of company that would 'fit' what the profile describes. Design your rubric with the understanding that it will be used by an evaluator to assess individual companies to assign a level of fit to the profile you will be shown, however this evaluator will not have access to the actual profile. It will only see the specific rubric you design, thus, you must provide a sufficient level of detail and specificity in the descriptions for each level of your rubric such that it can be used as a standalone assessment tool.

{additional_instruction}

You will structure your output as follows. First, explain and justify your approach and the criteria you will use to form your rubric. Next you will output on a new line the words "{rubric_delimiter}" followed by a line break. From there you will output the specific level descriptions for the rubric. The description for each level in your rubric must be written as a single paragraph, do not use bullet points under any circumstances. Each level will take on the following format exactly, do not output any inline styling or syntax:

Level N: Description

Ensure you always output five distinct levels in your rubric."""

PROFILE_RUBRIC_GENERATION_MAIN_PROMPT_STR = """Determine a scoring rubric for this by using information shown to you in the following profile '{profile}'. {additional_instruction}"""

PROFILE_RUBRIC_EXAMPLES_SYS_INSTRUCTION = """You will also be given will also be given a sampling of the set of stocks the rubric you generate will be used against. This sampling will contain a random set of stocks which may contain some stocks with a very strong fit to the profile and thus should score highly while other stocks included may be a weaker fit and would subsequently be scored lower. Each stock shown to you will also contain a summary containing information from that company relevant to the profile in question. Use this information to construct a high quality rubric that contains the depth and nuance required to sort stocks like the ones shown to you in the sampling into distinct levels. Keep in mind the sampling may not contain the best of the best or the worst of the worst, you must design your rubric to generalize beyond the sampling shown."""

PROFILE_RUBRIC_EXAMPLES_MAIN_INSTRUCTION = """Use the following sampling of stocks along with their stock summaries to help build your rubric. Each entry is seperated with '{delimiter}' and will contain the company name along with a summary:\n{samples}"""

RUBRIC_EVALUATION_MAIN_PROMPT = """The company in question is: {company_name}

Relation to the criteria: {reason}

Format your output as follows, a number indicating the level followed by ___ and a justification section explaining and justifying why you have chosen to assigned that particular level. Your justification must not make explicit reference to the specific level asigned though. Like so:

3___Here is some justification

The first thing you output must always be the number associated with the level selected. Do not include any inline styling or syntax, follow the format specified exactly. If you assign a level 0, there is no need to provide a justification but you must still output the ___."""

RUBRIC_EVALUATION_SYS_PROMPT = """Use the rubric below to assign a level for a given stock based on how well it fits the criteria. The rubric is shown below with a description for each level, ranging from 0 to 5:
{rubric_str}

Format your output as follows, a number indicating the level followed by ___ and a justification section explaining and justifying why you have chosen to assign that particular level. The first thing you output must always be the number associated with the level selected.
"""


PROFILE_RUBRIC_GENERATION_MAIN_OBJ = Prompt(
    name="PROFILE_RUBRIC_GENERATION_MAIN_PROMPT",
    template=PROFILE_RUBRIC_GENERATION_MAIN_PROMPT_STR,
)

PROFILE_RUBRIC_GENERATION_SYS_OBJ = Prompt(
    name="PROFILE_RUBRIC_GENERATION_SYS_PROMPT",
    template=PROFILE_RUBRIC_GENERATION_SYS_PROMPT_STR,
)

RUBRIC_EVALUATION_MAIN_OBJ = Prompt(
    name="RUBRIC_EVALUATION_MAIN", template=RUBRIC_EVALUATION_MAIN_PROMPT
)

RUBRIC_EVALUATION_SYS_OBJ = Prompt(
    name="RUBRIC_EVALUATION_SYS", template=RUBRIC_EVALUATION_SYS_PROMPT
)
