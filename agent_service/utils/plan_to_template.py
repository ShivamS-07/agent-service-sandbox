from agent_service.GPT.constants import DEFAULT_SMART_MODEL
from agent_service.GPT.requests import GPT
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.utils.async_db import AsyncDB, get_chat_history_from_db
from agent_service.utils.gpt_logging import chatbot_context
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.prompt_utils import Prompt

USER_TEMPLATE_CATEGORY = "My Templates"
TEMPLATE_FROM_PLAN_SYS_PROMPT = Prompt(
    name="TEMPLATE_FROM_PLAN_SYS_PROMPT",
    template=(
        "You are a prompt engineer tasked with creating a reusable template prompt from a given task plan. "
        "This plan contains tool calls generated based on client messages. "
        "Your goal is to summarize the client messages into a coherent and reusable prompt template. "
        "Follow the guidelines below to craft the template: "
        "\n\nGuidelines: "
        "\n- The prompt should be a coherent, concise summary of the client messages "
        "while maintaining the same tone and language. "
        "\n- Ensure the prompt is generalizable to similar tasks, using different tool input variables. "
        "\n- Use <code> for specific inputs such as topics, date ranges, portfolio names, sectors, index names, etc. "
        "\n- If user mentions 'my portfolio' or similar terms, and do not provide the exact name, "
        "use <code>my portfolio</code> tag in your prompt. "
        '\n- Use <option-dropdown type="stock" value="stock_name"></option-dropdown> tag ONLY for company names, '
        "or stock symbols. If a stock symbol or company name is mentioned multiple times, "
        "use the tag ONLY for the first mention, "
        "and then use 'target stock' or 'target company' or similar term for the rest without any tag. "
        "Make sure the type attribute is set to 'stock' and the value attribute is set to the given "
        "stock symbol, or company name. "
        "\n- If client mentioned any of these words such 'top' or 'bottom', 'high' or 'low', 'increase' or 'decrease', "
        "'positive' or 'negative', 'best' or 'worst', etc., use a <option-dropdown> tag to represent the pair. "
        "For example if client mentioned 'top', use "
        '<option-dropdown type="stat" options=\'["bottom", "top"]\' value="bottom"></option-dropdown> '
        "\n- The HTML tags <code> and <option-dropdown> should be embedded directly in the prompt "
        "to allow users to change values interactively. "
        "\n- Only use <code> and <optifon-dropdown> tags; avoid using any other HTML tags, markdown, "
        "or special characters. "
        "\n- If client already provided a list of options. Use all provided list with seperate tags. "
        "\n- Do not use any markdown formatting such as bold ('**'), italics ('*'), underline ('__'), newlines ('\\n')"
        ", or similar formatting in the prompt. It is forbidden to use '*' and '_' in the prompt. "
        "\n- The prompt should include all tool parameters/variables that are filled or interpreted directly "
        "based on the client messages. "
        "\n- Do not repeat a tool parameter/variable in the prompt if it is already mentioned in the prompt. "
        "For example, if the client message already mentions a specific stock symbol, "
        "do not repeat it in the prompt. "
        "\n- The final output should only include the generated prompt text, "
        "without any additional explanations or titles. "
        "\n- Here is a few examples of a template prompt based on given plan and user messages. "
        "Pay attention how the "
        "response template is formed based on the plan and user message:"
        "\n\nExample 1: "
        "\n##Execution Plan:\n"
        '1. spx_stocks = get_stock_universe(universe_name="SPX")  # Get stocks in the SPX index '
        '2. last_week_range = get_date_range(date_range_str="last week")  # Get the date range for the last week'
        "3. news_developments = get_all_news_developments_about_companies(stock_ids=spx_stocks, "
        "date_range=last_week_range)  # Get news developments for SPX stocks over the last week "
        "4. news_summary = summarize_texts(texts=news_developments, "
        'topic="US election poll predictions and market impact")  '
        "# Summarize news developments focusing on US election poll predictions and market impact "
        "5. output1 = prepare_output(object_to_output=news_summary, "
        'title="Summary of news on US election poll predictions and market impact")  '
        "# Output the summary of news "
        '6. portfolio_id = convert_portfolio_mention_to_portfolio_id(portfolio_name="my portfolio")  '
        "# Convert portfolio mention to portfolio ID"
        "7. portfolio_holdings = get_portfolio_holdings(portfolio_id=portfolio_id)  # Get the portfolio holdings "
        '8. output2 = prepare_output(object_to_output=portfolio_holdings, title="Portfolio Holdings")  '
        "# Output the portfolio holdings "
        '9. nvidia_stock_id = stock_identifier_lookup(stock_name="Nvidia")  # Get Nvidia stock identifier '
        "10. nvidia_news = get_all_news_developments_about_companies(stock_ids=[nvidia_stock_id], "
        "date_range=last_week_range)  "
        "# Get news developments for Nvidia over the last week "
        '11. nvidia_analysis = summarize_texts(texts=nvidia_news, topic="Nvidia price impact by US election result")  '
        "# Analyze Nvidia price impact by US election result "
        '12. output3 = prepare_output(object_to_output=nvidia_analysis, title="Analysis of Nvidia price impact by '
        'US election result")  # Output the analysis of Nvidia price impact '
        "\n##User Messages:\n"
        "Client: Write a summary of news in the last week focusing on topics US election poll predictions and its "
        "impact on the market (SPX index). In separate a section, show my portfolio holding. "
        "In another section include a brief analysis of how Nvidia prices are impacted by the US election result. "
        "\n##Prompt Template: "
        "Write a summary of news in the <code>last week</code> focusing on topics "
        "<code>US election poll predictions</code> "
        "and its impact on the market <code>SPY</code> index. "
        "In a separate section, show <code>my portfolio</code> holding. "
        'In a new section include a brief analysis of how <option-dropdown type="stock" value="Nvidia"> '
        "</option-dropdown> prices are impacted by the US election result."
        "\n\nExample 2: "
        "\n##Execution Plan:\n"
        '1. spy_stocks = get_stock_universe(universe_name="SPY")  # Get stocks in the SPY universe '
        '2. percentage_gain_table = get_statistic_data_for_companies(statistic_reference="percentage gain '
        'over the past year", stock_ids=spy_stocks)  # Get percentage gain over the past year for SPY stocks '
        "3. bottom_100_stocks_table = transform_table(input_table=percentage_gain_table, transformation_description="
        '"Filter to bottom 100 stocks by annual gain")  # Filter to bottom 100 stocks by annual gain '
        "4. bottom_100_stocks = get_stock_identifier_list_from_table(input_table=bottom_100_stocks_table)  "
        "# Extract stock IDs from the bottom 100 stocks table "
        '5. pe_ratio_table = get_statistic_data_for_companies(statistic_reference="P/E", stock_ids=bottom_100_stocks)'
        " #Get P/E ratio for the bottom 100 stocks "
        "6. filtered_pe_stocks_table = transform_table(input_table=pe_ratio_table, transformation_description "
        '="Filter stocks with P/E ratio greater than 10 and less than 25")  '
        "# Filter stocks with P/E ratio between 10 and 25 "
        "7. filtered_pe_stocks = get_stock_identifier_list_from_table(input_table=filtered_pe_stocks_table)  "
        "# Extract the stocks IDs from the filtered P/E stocks table "
        "8. positive_news_stocks = get_stock_recommendations(stock_ids=filtered_pe_stocks, "
        'filter=True, buy=True, news_horizon="3M", news_only=True)  '
        "# Filter stocks with positive news over the past 3 months"
        "9. top_3_stocks = get_stock_recommendations(stock_ids=positive_news_stocks, filter=True, buy=True, "
        "num_stocks_to_return=3)  # Get top 3 stocks with positive news "
        "10. output_top_3_stocks = prepare_output(object_to_output=top_3_stocks, "
        'title="Top 3 stocks with positive news")  # Output the top 3 stocks with positive news '
        "11. top_3_texts = get_default_text_data_for_stocks(stock_ids=top_3_stocks)  "
        "# Get default text data for the top 3 stocks "
        "12. deep_dive_analysis = per_stock_summarize_texts(stocks=top_3_stocks, texts=top_3_texts, "
        'topic="Deep dive analysis")  # Create a deep dive analysis for each of the top 3 stocks '
        "13. output_deep_dive_analysis = prepare_output(object_to_output=deep_dive_analysis, title="
        '"Deep dive analysis on top 3 stocks")  # Output the deep dive analysis on the top 3 stocks '
        "\n##User Messages:\n"
        "Client: Look at the bottom 100 stocks in SPY based on their percentage gain over the past year. "
        "Then filter that down to companies with a PE > 10 and PE < 25. Then filter those companies down to "
        "good buys with positive news over the past 3 months. Then add a deep dive into the top 3 ideas "
        "\n##Prompt Template: "
        'Analyze the <option-dropdown type="stat" options=\'["bottom", "top"]\' value="bottom"></option-dropdown> '
        "<code>100</code> stocks in the <code>SPY</code> universe based on "
        "<code>the percentage gain over the past year</code>. Filter these companies to those with a <code>P/E</code> "
        "ratio greater than <code>10</code> and less than <code>25</code>. "
        "Further narrow down these companies to those considered good buys with "
        '<option-dropdown type="stat" options=\'["positive", "negative"]\' value="positive"></option-dropdown> '
        "news over <code>the past 3 months</code>. Provide a deep dive analysis into the "
        '<option-dropdown type="stat" options=\'["bottom", "top"]\' value="bottom"></option-dropdown> '
        "<code>3</code> stocks identified as having "
        '<option-dropdown type="stat" options=\'["positive", "negative"]\' value="positive"></option-dropdown>'
        " news."
    ),
)


TEMPLATE_FROM_PLAN_MAIN_PROMPT = Prompt(
    name="TEMPLATE_FROM_PLAN_MAIN_PROMPT",
    template=(
        "You have been provided with a task plan generated based on client messages. "
        "Your goal is to generate a reusable template prompt based on the given plan and user messages "
        "and follow all the guidelines provided in the system prompt. "
        "\n##Plan:"
        "\n{plan}\n"
        "\n##User Messages: "
        "\n{chat}\n"
        "Now proceed to generate the template prompt."
    ),
)


@async_perf_logger
async def generate_template_from_plan(
    plan: ExecutionPlan,
    agent_id: str,
    db: AsyncDB,
    model: str = DEFAULT_SMART_MODEL,
) -> str:
    chat_history = await get_chat_history_from_db(agent_id=agent_id, db=db)

    sys_prompt = TEMPLATE_FROM_PLAN_SYS_PROMPT.format()
    main_prompt = TEMPLATE_FROM_PLAN_MAIN_PROMPT.format(
        plan=plan.get_formatted_plan(numbered=True),
        chat=chat_history.get_gpt_input(client_only=True),
    )

    llm = GPT(context=chatbot_context(agent_id), model=model)
    prompt_str = await llm.do_chat_w_sys_prompt(main_prompt, sys_prompt)

    return prompt_str
