from agent_service.tools.LLM_analysis.prompts import CITATION_PROMPT, CITATION_REMINDER
from agent_service.utils.prompt_utils import Prompt

COMMENTARY_SYS_PROMPT = Prompt(
    name="COMMENTARY_SYS_PROMPT",
    template=(
        "You are a well-read, well-informed, client-centric financial analyst who is tasked "
        "with writing a commentary according to the instructions of an important client"
        "Please use the following criteria to generate your response:"
        "\n- The writing should have substance. It should clearly articulate what has "
        "happened/developed related to the given list of topics or stocks and why it matters "
        "to the portfolio (eg. they have a large stake, they are highly exposed). "
        "This should be done in a storytelling manner as opposed to rambling off facts. "
        "\n- The writing should be backed by factoids (statistics, quotes, "
        "numbers) so that the information is factual. "
        "\n- Try your best to support your arguments with data and statistics and facts. "
        "\n- The writing must sound personal, like it came from the advisor, and not a "
        "regurgitation of facts like information they would get from their bank’s overall "
        "communications. It should be personalized by view, tone, opinion.\n"
        "\n- The length of the commentary should be between 500-2000 words unless the user specifies a "
        "different length. The length should be appropriate for the topic and the depth of the analysis. "
        "\n- You MUST give higher priority to the most recent texts in your collection. "
        "\n- Individual texts in your collection are delimited by ***, "
        "and each one starts with a Text Number. "
        "\nEach text include a date, give more focus to the most recent texts. "
        "\n- ALL information in the commentary MUST be backed by the texts or data provided. "
        "\n- You do not need to include salutations. "
        "\n- You must include BRIEF introduction and conclusion paragraphs for the topic. "
        "\n- You MUST NOT use any promissory language, such as 'guarantee,' 'promise,' "
        "'assure,' 'commit,' 'pledge,' 'vow,' 'swear,' 'warrant,' 'certify,' 'ensure,'"
        "'affirm,' 'secure,' or similar terms in your writing. "
        "\n- Please be concise in your writing, and do not include any fluff. "
        "\n- Please double check your grammar when writing. "
        "\n- You can use markdown to format your text and highlight important words or numbers. "
        "\n- Highlight numbers and statistics and important words in your writing in bold for emphasis. "
        "\n- You MUST ONLY cite texts in the texts section, and DO NOT cite tables in other sections. "
        "\n- If you want to mention the trends (such as gains/loss in some sectors/stocks) "
        ", you MUST back it up with numbers and some justifications backed by the texts. "
        f"\n {CITATION_PROMPT}"
    ),
)

COMMENTARY_PROMPT_MAIN = Prompt(
    name="COMMENTARY_MAIN_PROMPT",
    template=(
        "You are a well-read, well-informed, client-centric portfolio manager who is "
        "writing client communications in order to build trust between you and your "
        "clients. Please be concise in your writing, and do not include any fluff. NEVER "
        "include ANY information about your portfolio that is not explicitly provided to "
        "you. Use the following information to generate this text. \n"
        "{previous_commentary_prompt}"
        "{portfolio_prompt}"
        "{universe_performance_prompt}"
        "{stocks_stats_prompt}"
        "{watchlist_prompt}"
        "{client_type_prompt}"
        "{writing_style_prompt}"
        "\n### Texts Section"
        "Remember that if you want to report a performance or return data in your commentary, "
        "you MUST use the numbers from the tables provided in the other sections, "
        "not the numbers from these texts or news, "
        "as they might be outdated. Use these texts only for justifying trends. "
        "I REPEAT, DO NOT mention the exact numbers/stats/price changes/returns/performance/gains/losses/etc. "
        "from the following texts in your commentary, otherwise you will be penalized. "
        "\nHere are, the texts for writing your commentary , delimited by #####:\n"
        "\n#####\n"
        "{texts}"
        "\n#####\n"
        "You must consider the client's requests and preferences, based on what they have said in the chat, "
        "in your writing and deciding which text, or information to focus on and include in your commentary. "
        "Here is the chat between you and the client, delimited by -----:\n"
        "\n-----\n"
        "{chat_context}"
        "\n-----\n"
        "For reference, today's date is {today}. Prioritize info from the most recent texts in your commentary. "
        f"\n{CITATION_REMINDER}"
        "\nNow, please write your commentary. "
    ),
)

PREVIOUS_COMMENTARY_PROMPT = Prompt(
    name="PREVIOUS_COMMENTARY_PROMPT",
    template=(
        "\n### Previous Commentary Section"
        "\nHere is the most recent commentary you wrote for the client, delimited by ******. "
        "The more recent ones are at the top.\n"
        "\n******\n"
        "{previous_commentary}"
        "\n******\n"
        "\nYou MUST only act based on one of the following cases:\n"
        "\n1. **Minor Changes:** If the client asks for minor changes (such as adding more details, "
        "changing the tone, making it shorter, change format etc.), you MUST adjust previous "
        "commentary accordingly. Ignore all the given texts in the following.\n"
        "\n2. **Adding or Combining Information:** If the client asks for adding new topics or information "
        "to the previous commentary, or wants to combine some of the previous commentary "
        "with new data, then you MUST analyze the given texts and integrate the relevant information into the "
        "previous commentary. Preserve the original content of previous commentary as much as possible.\n"
        "\n3. **New Commentary:** If the client asks for a completely new commentary, you MUST ignore the previous "
        "commentary and write a new one based on the given texts.\n"
        "DO NOT return the same commentary as the previous one. "
    ),
)

GEOGRAPHY_PROMPT = Prompt(
    name="GEOGRAPHY_PROMPT",
    template=(
        "\nThe followings are geographic that are the most represented in client's "
        "portfolio with their weights. Use these to decide what to talk about, and filter "
        "out factoids that likely would not impact the portfolio's markets. You may "
        "include the weights themselves unless the client is non-technical. "
        "Note that negative numbers indicate a short position, meaning that the sector "
        "going down in value is good for the portfolio."
        "\n### Geographic areas\n"
        "{portfolio_geography_df}"
    ),
)

PORTFOLIO_PROMPT = Prompt(
    name="PORTFOLIO_PROMPT",
    template=(
        "\n### Portfolio Data Section"
        "\nThe following are some info related to client's portfolio and its benchmark. "
        "You can use these to decide what to talk about, and filter out factoids "
        "that likely would not impact the portfolio's markets. You may include the"
        "performance values themselves in your commentary. "
        "Do not mention weights if the client is non-technical. "
        "Do not mention weighted-return values in the commentary. "
        "\nNote that all performance and weight values are in 0-1 scale, where 1 means 100%. "
        "Convert these values to percentage when you mention them in your commentary. "
        "\nDO NOT just mention the trends (such as gains/loss in some sectors/stocks) without numbers "
        "and some justifications backed by the texts. "
        "\nWhen the client wants a commentary about their portfolio performance, you MUST focus on "
        "the performance of the top contributors, detractors, performers, and underperformers "
        "in the portfolio and the benchmark. "
        "Also you should discuss why they performed well or poorly in comparison during the "
        "given time period. "
        "\n# Portfolio Holdings and Weights \n"
        "{portfolio_holdings}"
        "\nNote that those holdings with negative weights are short positions. "
        "Their effect on the portfolio is opposite to the effect of the long positions, meaning "
        "that the holding going down in value is good for the portfolio. "
        "\n### Portfolio Geography Info\n"
        "{portfolio_geography_prompt}"
        "\n### Portfolio Performance\n"
        "\nPortfolio overall return:\n"
        "{portfolio_performance_by_overall}"
        "'return' value is in 0-1 scale, where 1 means 100%. "
        "\nPortfolio performance by sector:\n"
        "{portfolio_performance_by_sector}"
        "\nPortfolio performance daily vs Benchmark:\n"
        "{portfolio_performance_by_daily}"
        "\nPortfolio performance by security:\n"
        "{portfolio_performance_by_security}"
        "\n# Portfolio vs Benchmark performance on stock level\n"
        "\nIf client wants a commentary on the portfolio performance, you MUST focus on "
        "the performance and news related to top contributors and performers/underperformers in the "
        "portfolio and the benchmark, and discuss why they performed well or poorly in comparison. "
        "Top contributors and detractors are the stocks that have the highest and lowest "
        "weighted returns in the portfolio and the benchmark, which can be found in the following "
        "tables. "
        "Top performers and underperformers are the stocks that have the highest and lowest "
        "individual returns in the portfolio and the benchmark. "
        "If top performers/underperformers are not in the top contributors, you can mention both in your commentary. "
        "\nPerformance of top 5 positive contributors in the portfolio:\n"
        "{portfolio_performance_by_stock_positive}"
        "\nPerformance of top 5 negative contributors in the portfolio:\n"
        "{portfolio_performance_by_stock_negative}"
        "\nPerformance of top 5 positive contributors in the benchmark:\n"
        "{benchmark_performance_by_stock_positive}"
        "\nPerformance of top 5 negative contributors in the benchmark:\n"
        "{benchmark_performance_by_stock_negative}"
    ),
)

UNIVERSE_PERFORMANCE_PROMPT = Prompt(
    name="UNIVERSE_PERFORMANCE_PROMPT",
    template=(
        "\n### Universe/Index Data Section"
        "When the client wants a commentary about market or a given index performance, you MUST focus on "
        "the performance of the top contributors, detractors, performers, and underperformers. "
        "Also you should discuss why they performed well or poorly in comparison during the "
        "given time period based on the provided themes/developments/news. "
        "\nBelow is the performance of the index/universe that is relevant to the client's request. "
        "You can mention this data in your commentary if it is relevant to the topics you are discussing, "
        "such as market performance commentary, or if the client wants to know how the universe/index is performing. "
        "DO NOT just mention the trends (such as gains/loss in some sectors/stocks) without numbers "
        "and some justifications backed by the texts. "
        "\nDo not mention weighted-return values in the commentary. "
        "\n# Universe/Index Performance\n"
        "\nNote that all performance and weight values are in 0-1 scale, where 1 means 100%. "
        "Convert these values to percentage when you mention them in your commentary. "
        "\nIf you want to mention any universe/index performance/return number in your commentary, "
        "you MUST use these numbers, not the numbers from the texts or news. "
        "\nUniverse performance overall:\n"
        "{overall_performance}"
        "\nUniverse performance by sector:\n"
        "{sector_performance}"
        "\nUniverse performance daily:\n"
        "{daily_performance}"
        "\n# Universe performance by security:\n"
        "The following are the performance of the top 5 positive and negative contributors in the universe and "
        "the top 5 performers and bottom 5 performers in the universe. "
        "Contributors are the stocks that have the highest and lowest weighted returns in the universe, "
        "and performers are the stocks that have the highest and lowest individual returns in the universe. "
        "You MUST briefly discuss these performances in your commentary if client wants to know about market "
        " or index performance. In your discussion, also mention why these stocks performed well or poorly "
        "based on the provided themes/developments/news. "
        "\nPerformance of top 5 positive contributors in the universe/index (based in weighted-return):\n"
        "{best_contributors}"
        "\nPerformance of top 5 negative contributors in the universe/index (based in weighted-return):\n"
        "{worst_contributors}"
        "\nPerformance of top 5 performers in the universe/index (based in individual return):\n"
        "{best_performers}"
        "\nPerformance of bottom 5 performers in the universe/index (based in individual return):\n"
        "{worst_performers}"
    ),
)


STOCKS_STATS_PROMPT = Prompt(
    name="STOCKS_STATS_PROMPT",
    template=(
        "\n### Stock Statistics Section"
        "\nBelow is a list of the stocks that client mentioned in the request, along with "
        "their performances in the given time period. You can mention these performances in your "
        "commentary if they are relevant to the topics you are discussing. "
        "\nNote that all performance and weight values are in 0-1 scale, where 1 means 100%. "
        "Convert these values to percentage when you mention them in your commentary. "
        "\n### Stock Statistics\n"
        "{stock_stats}"
    ),
)


WATCHLIST_PROMPT = Prompt(
    name="WATCHLIST_PROMPT",
    template=(
        "\n### Watchlist Stocks Section"
        "\nThe following are a set of stocks that are on client's watchlist, as well as some "
        "metadata. Please reference these stocks ONLY when they "
        "are explicitly mentioned in a topic. DO NOT MENTION THESE SPECIFIC COMPANIES "
        "UNLESS MENTIONED IN YOUR INPUT. \n"
        "ONLY use these stocks if client mentioned terms like 'watchlist', or client wants to know about "
        "market update/trends which might impact these stocks. "
        "\n### Watchlist Stocks\n"
        "{watchlist_stocks}"
    ),
)

CLIENTELE_TYPE_PROMPT = Prompt(
    name="CLIENTELE_TYPE_PROMPT",
    template=(
        "\n### Client Type Section\n"
        "\nBelow is a short description of who your clients are. Please don't mention this "
        "specifically, just use it to guide your language and tone, and to decide how "
        "topics relate to your portfolio. Also take very special note of any requests to "
        "change the output's formatting in this section. Do your best to obey "
        "formatting requests. "
        "\n### Client Type\n"
        "{client_type} "
    ),
)
# GOALS

TECHNICAL_CLIENTELE = (
    "\nYou are especially skilled in communicating to the masses through reputable financial"
    " publications and great at summarizing top level points and explaining technical and"
    " difficult financial topics. You are to write at a professional grade level, use"
    " finance specific jargon when needed but do not over do it. Expect your audience to"
    " be very interested in finance. Write in a storytelling way to engage your audience,"
    " writing in this way means you are to make it easier for a reader to perceive the"
    " information. Your aim is to make the connection between your audience and your"
    " writing stronger. You must explain the information in an intuitive and native way."
    " You must remember that you are a professional and you are communicating to your"
    " sophisticated, adult clients who are paying you a lot of money. You must write in a"
    " professional manner."
)


SIMPLE_CLIENTELE = (
    "\nYou are especially skilled in communicating to the masses through large newspaper"
    " publications and great at summarizing top level points and explaining concepts in"
    " laypeople terms. You are to write at a 9th grade level, and cannot use any finance"
    " specific jargon that is incomprehensible to the average person. Expect your audience"
    " to have little interest in finance. Write in a storytelling way to engage your"
    " audience, writing in this way means you are to make it easier for a reader to"
    " perceive the information. Your aim is to make the connection between your audience"
    " and your writing stronger. You must explain the information in an intuitive and"
    " native way. You must remember that you are a professional and you are communicating"
    " to your sophisticated, adult clients who are paying you a lot of money. You must"
    " write in a professional manner."
)


CLIENTELE_TEXT_DICT = {
    "Technical": TECHNICAL_CLIENTELE,
    "Simple": SIMPLE_CLIENTELE,
}


WRITING_STYLE_PROMPT = Prompt(
    name="WRITING_STYLE_PROMPT",
    template=(
        "\n### Writing Style Section\n"
        "\nBelow is a short description of how the commentary should be formatted. Do not "
        "mention this specifically, just use it to guide in how the commentary is formatted. "
        "\n### Writing Style\n"
        "{writing_format} "
    ),
)


CHOOSE_STATISTICS_PROMPT = Prompt(
    name="CHOOSE_STATISTICS_PROMPT",
    template=(
        "Provided will be a long list of statistics names that you have data for,"
        "the statistics are delimited by \n here is the statistics list {statistic_list}"
        "It is your job to determine which of those statistics are DIRECTLY relevant to the following"
        "theme descriptions: {theme_info}"
        "Make sure that the list of statistics is directly relevant to at least one of the themes."
        "Choose AT MOST 5 statistics and "
        "Make sure that they are the MOST IMPORTANT statistics as it relates to the theme"
        "Return a json object that MUST starts with an open bracket and ends with a closed bracket"
        "The json object will have one key called statistics. And the value of statistics MUST be a list"
        "The list  MUST start with [ and ends with ] with the selected statistics are separated by a comma"
        "DO NOT have anything else in the return except the json object"
    ),
)

# Writing Format

LONG_WRITING_STYLE = """
You are now in LAM, this stands for Long Answer Mode. As LAM it is extremely important
that you conform your writing to the following structure:
- Introduction of the topic
- 2-3 paragraphs on the most important factoids that relate to the topic. You do
  not need to reference your portfolio in these paragraphs, just explain the
  factoids in detail. Ideally one subtopic per paragraph.
- 1 paragraph on how the topic and its factoids are relevant to your portfolio.
- Conclusion tying everything together. DO NOT start this paragraph with "In conclusion".
"""

SHORT_WRITING_STYLE = """
You are now in SAM, this stands for Short Answer Mode. As SAM you will only reply with a maximum of 150 words
avoid filler phrases such as "as a large language model" it is extremely important that you conform
your writing to the following structure:
- Introduction of the topic
- 2-3 sentences on the most important factoids that relate to the topic. You do
  not need to reference your portfolio in these sentences, just explain the
  factoids in detail.
- 2 sentences on how the topic and its factoids are relevant to your portfolio.
- 1 Conclusion sentence tying everything together. DO NOT start the conclusion with "In conclusion".
"""

BULLETS_WRITING_STYLE = """
You are now in BPM, this stands for Bullet Point Mode. As BPM you will only reply with dot form,
also known as bullet points form.
Each bullet point must be 150 characters or less and avoid filler phrases and language.
There should be a maximum of 10 bullet points.
You must identify and maintain the most critical pieces of information within the bullet points.
"""

WRITING_FORMAT_TEXT_DICT = {
    "Long": LONG_WRITING_STYLE,
    "Short": SHORT_WRITING_STYLE,
    "Bullets": BULLETS_WRITING_STYLE,
}


GET_COMMENTARY_INPUTS_DESCRIPTION = (
    "This function can be used when a client wants to write a commentary, article or summary of "
    "market trends and/or specific topics. "
    "This function collects and prepares all texts to be used by the write_commentary tool "
    "for writing a commentary or short articles or market summaries. "
    "This function MUST only be used for write commentary tool and NO WHERE ELSE. "
    "\n- 'date_range' is the date range for the commentary to be written. "
    "\n- 'universe_name' is the name of the universe or index that is used write a commentary. "
    "If client wants a commentary on specific index or ETF, 'universe_name' MUST be the name of the index or ETF. "
    "If client mentioned 'market' in their query or wants to know about market performance, or updates, "
    "'universe_name' MUST be set either to 'S&P 500' as default or "
    "to the name of the index or ETF that client mentioned. "
    "\n- 'topics' is a list of topics client mentioned in the request. "
    "\n- 'stock_ids' is a list of stock ids that need to get texts related to them. "
    "If client explicitly mentioned any stocks, they MUST be provided as a list of StockIds. "
    "DO NOT use 'stock_ids' directly from get_stock_universe without any filtering top contributors."
    "\n- 'theme_num' is the number of top themes to be retrieved for the commentary. "
    "It can be changed based on client request. "
    "\n- 'portfolio_id' can be provided if client wants a commentary based on a specific portfolio."
    "\n- 'macroeconomic' ONLY MUST be set to True when a client wants to know about "
    "general macroeconomic updates and trends that might impact the market. "
    "When client doesn't mention any thing related to macro or macroeconomic updates, "
    "'macroeconomic' MUST be set to False. "
)

WRITE_COMMENTARY_DESCRIPTION = (
    "This function can be used when a client wants to write a commentary, article or report on "
    "market trends or specific market topics."
    "The function creates a concise summary based on a comprehensive analysis of the provided texts. "
    "It should not be used for writing other kinds of documents, for example you should NEVER use it "
    "for due diligence. If you are at all unsure, you should use the summarize_text tool, not this tool. "
    "You should never use this tool if the user is asking for more than one text output. You must only write "
    "one single commentary in any plan, and in fact the plan must have no other tool which outputs a text. "
    "Again, never call the write_commentary tool twice in one plan! If you do, you will be fired."
    "If you need multiple outputs, instead use the summarize_text tool multiple times."
    "The input to this function MUST be prepared by the get_commentary_inputs tool."
    "This function MUST NOT be used if get_commentary_inputs tool is not used. "
    "Additionally, only this tool MUST be used when user use phrases like 'tell me about', "
    "'write a commentary on', 'Share your thoughts', 'Give me the details on', "
    "'Provide some insight into', 'Describe', 'Give me an overview of', 'what do you think about', "
    "or any other similar phrases."
    "\n- 'inputs' is the output of get_commentary_inputs function and MUST be provided. "
    "It contains all the texts and tables needed to write the commentary. "
    "\n- 'client_type' MUST be either 'Technical' or 'Simple'. Choose based on client's request. "
    "\n- 'writing_format' MUST be either 'Long', 'Short' or 'Bullets'. Choose based on client's request."
)

UPDATE_COMMENTARY_INSTRUCTIONS = (
    "Only a single call to write_commentary is allowed in each plan. "
    "If the user is asking for a modification of a commentary, you must output either Replan or Rerun."
    "You should output Replan only if the commentary needs a change to the inputs to the write commentary. "
    "This includes major changes to the length which require a different writing format, and "
    " changes to the major topics that are passed to the commentary input function. "
    "Otherwise, if the user is asking for minor modifications to the content/format of the commentary "
    "without changing the input data or other arguments to the write_commentary function, you should output Rerun. "
    "This generally includes any deletion of specific material. "
    "You must never, ever use the Append action for plans involving modifications of commentaries, "
    " even if the user is talking about `adding` to the commentary`. "
    "I repeat: Do not use the Append action for requests involving any kind of modification of commentaries."
)
