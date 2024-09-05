# flake8: noqa
CANNED_PROMPTS = [
    {
        "id": "write_commentary",
        "prompt": "Write a commentary on market performance over the last month. Back observations up with data. Format the commentary to make it easy to read.",
    },
    {
        "id": "identify_top_tech_stocks",
        "prompt": "Identify the top 5 performing stocks in the technology sector over the past year, focusing on companies with a market cap above $10B. Provide a summary of their recent financial performance and any significant news.",
    },
    {
        "id": "summarize_costco",
        "prompt": "Summarize all the major developments for COST over the past year. Focus your analysis on corporate filings and earnings calls. Show the developments in point form as a timeline with dates. Bold anything important. For each development mention if it is positive or negative and why it is significant to COST.",
    },
    {
        "id": "summarize_netflix",
        "prompt": "Give me all information available about AAPL that will affect its upcoming earnings call, use only top news sources, refer to past earnings transcripts. Show answer in bullet points.",
    },
    {
        "id": "spy_pe_news_filter",
        "prompt": "Look at the bottom 100 stocks in SPY based on their percentage gain over the past year. Then filter that down to companies with a PE > 10 and PE < 25. Then filter those companies down to good buys with positive news over the past 3 months. Then add a deep dive into the top 3 ideas",
    },
]
