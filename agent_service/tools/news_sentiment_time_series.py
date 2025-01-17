import datetime
from typing import List, Optional, Tuple

import pandas as pd

from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable, TableColumnMetadata
from agent_service.tool import ToolArgs, ToolCategory, default_tool_registry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.postgres import get_psql


class GetNewsSentimentTimeSeriesInput(ToolArgs):
    stock_ids: List[StockID]
    date_range: Optional[DateRange] = None


async def _get_news_sentiment_helper(
    stock_ids: List[StockID], date_range: DateRange
) -> List[Tuple[StockID, datetime.date, float]]:
    sql_query = """
    SELECT
        gbi_id,
        jsonb_agg(elem) AS sentiments
    FROM (
        SELECT
            gbi_id,
            elem
        FROM
            nlp_service.stock_news_updates,
            jsonb_array_elements(sentiment_history::jsonb) AS elem
        WHERE
            gbi_id = ANY(%(gbi_ids)s) AND
            (elem->>0)::date BETWEEN %(start_date)s AND %(end_date)s
    ) AS filtered_data
    GROUP BY gbi_id;
    """
    rows = get_psql().generic_read(
        sql_query,
        {
            "gbi_ids": list(map(lambda stock: stock.gbi_id, stock_ids)),
            "start_date": date_range.start_date if date_range.start_date else datetime.date.min,
            "end_date": date_range.end_date if date_range.end_date else datetime.date.max,
        },
    )

    stock_ids_map = {sid.gbi_id: sid for sid in stock_ids}

    output = []
    for row in rows:
        stock_id = stock_ids_map[row["gbi_id"]]
        for elem in row["sentiments"]:
            output.append((stock_id, elem[0], elem[1]))
    return output


@tool(
    description=(
        "This tool helps generate a time series of news sentiment for one or more stocks."
        " Use this when a only when time series is required (rather than a single value)"
        " You must ONLY use this client when a user wants to see a sentiment graph, if the client wants"
        " individual, current sentiment values for a particular , use the get_stock_recommendation tool!"
        " I repeat, do NOT use this tool unless the user wants a graph of sentiment."
        " When a user wants a sentiment analysis of a stock, use the get_stock_recommendations tool."
        " This tool must NEVER, EVER be used to get highest or lowest sentiment scores for a set of"
        " stocks, use the get_stock_recommendations tool instead."
        " This tool must NEVER be used to filter stocks by sentiment score, use get_stock_recommendations instead"
        " News Sentiment is the same as investor sentiment, public sentiment."
        " The input is a list of stock IDs and an optional Date Range."
        " If the date_range is not provided, it will be assumed to be start_date = 1 year before today,"
        " end date should be the current date."
        " The output is a StockTable with columns: Security, Date, Sentiment."
        " The output StockTable must be converted to a Line Graph for visualization."
    ),
    category=ToolCategory.STOCK_SENTIMENT,
    tool_registry=default_tool_registry(),
)
async def get_news_sentiment_time_series(
    args: GetNewsSentimentTimeSeriesInput, context: PlanRunContext
) -> StockTable:
    date_range = args.date_range
    if not date_range:
        current = datetime.datetime.now().date()
        past = datetime.date(current.year - 1, current.month, current.day)
        date_range = DateRange(start_date=past, end_date=current)

    sentiments = await _get_news_sentiment_helper(args.stock_ids, date_range)
    df = pd.DataFrame(sentiments, columns=["Security", "Date", "Sentiment"])
    table = StockTable.from_df_and_cols(
        data=df,
        columns=[
            TableColumnMetadata(label="Security", col_type=TableColumnType.STOCK),
            TableColumnMetadata(label="Date", col_type=TableColumnType.DATE),
            TableColumnMetadata(label="Sentiment", col_type=TableColumnType.FLOAT),
        ],
    )
    return table
