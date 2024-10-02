import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL, DEFAULT_EMBEDDING_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import (
    NewsPoolArticleText,
    StockNewsDevelopmentArticlesText,
    StockNewsDevelopmentText,
)
from agent_service.planner.errors import EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.date_utils import get_now_utc, parse_date_str_in_utc
from agent_service.utils.postgres import Postgres, get_psql
from agent_service.utils.prompt_utils import Prompt

EMBEDDING_POOL_BATCH_SIZE = 100
MIN_POOL_PERCENT_PER_BATCH = 0.1
MAX_NUM_RELEVANT_NEWS_PER_TOPIC = 200


@dataclass(frozen=True)
class NewsTopic:
    topic_id: str
    first_article: datetime.datetime
    last_article: datetime.datetime
    major_updates: List[datetime.datetime]


async def _get_news_developments_helper(
    stock_ids: List[StockID],
    user_id: str,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
) -> Dict[StockID, List[StockNewsDevelopmentText]]:

    # Response now has a list of topics. Build an association dict to ensure correct ordering.
    stock_to_topics_map: Dict[StockID, List[NewsTopic]] = defaultdict(list)
    gbi_id_to_stock_map: Dict[int, StockID] = {stock.gbi_id: stock for stock in stock_ids}

    if not start_date:
        start_date = (get_now_utc() - datetime.timedelta(days=7)).date()
    if not end_date:
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = get_now_utc().date() + datetime.timedelta(days=1)

    start_dt = datetime.datetime(
        year=start_date.year,
        month=start_date.month,
        day=start_date.day,
        tzinfo=datetime.timezone.utc,
    )
    end_dt = datetime.datetime(
        year=end_date.year,
        month=end_date.month,
        day=end_date.day,
        hour=23,
        minute=59,
        second=59,
        tzinfo=datetime.timezone.utc,
    ) + datetime.timedelta(
        hours=12
    )  # add extra time to prevent timezone weirdness

    sql = """
            WITH t AS (
              SELECT
                snt.topic_id,
                CASE WHEN snt.gbi_id = -1 THEN sntm.gbi_id ELSE snt.gbi_id END AS gbi_id,
                snt.topic_descriptions,
                snt.created_at
              FROM nlp_service.stock_news_topics snt
              LEFT JOIN nlp_service.stock_news_topic_map sntm ON snt.topic_id = sntm.topic_id
            )
            SELECT t.topic_id::TEXT, t.gbi_id, t.topic_descriptions,
                MIN(sn.published_at) AS first_article_date,
                MAX(sn.published_at) AS last_article_date
            FROM nlp_service.stock_news sn
            JOIN t ON sn.topic_id = t.topic_id
            WHERE t.gbi_id = ANY(%(gbi_ids)s) AND sn.topic_id NOTNULL
            GROUP BY t.topic_id, t.gbi_id, t.topic_descriptions
            HAVING MAX(sn.published_at) >= %(start_date)s AND MIN(sn.published_at) <= %(end_date)s
    """
    rows = get_psql().generic_read(
        sql,
        {
            "gbi_ids": list(gbi_id_to_stock_map.keys()),
            "start_date": start_dt,
            "end_date": end_dt,
        },
    )

    for row in rows:
        stock = gbi_id_to_stock_map[row["gbi_id"]]
        stock_to_topics_map[stock].append(
            NewsTopic(
                topic_id=row["topic_id"],
                first_article=row["first_article_date"],
                last_article=row["last_article_date"],
                major_updates=list(
                    sorted((parse_date_str_in_utc(desc[1]) for desc in row["topic_descriptions"]))
                ),
            )
        )

    output_dict: Dict[StockID, List[StockNewsDevelopmentText]] = {}
    for stock in stock_ids:
        topics = stock_to_topics_map[stock]
        if not topics:
            continue
        topic_list = []
        for topic in topics:
            major_updates_in_range = [
                mu for mu in topic.major_updates if mu >= start_dt and mu <= end_dt
            ]
            if start_dt <= topic.first_article and end_dt >= topic.last_article:
                # If the topic falls FULLY within the date range, we can just
                # use the development text as-is (i.e. the description)
                topic_list.append(
                    StockNewsDevelopmentText(
                        id=topic.topic_id, stock_id=stock, timestamp=topic.last_article
                    )
                )
            elif major_updates_in_range:
                # Otherwise, if some major update to the topic happened within
                # the date range, use an article between the range's start and
                # the major update's timestamp.
                topic_list.append(
                    StockNewsDevelopmentText(
                        id=topic.topic_id,
                        stock_id=stock,
                        timestamp=major_updates_in_range[-1],
                        only_get_articles_start=start_dt,
                        only_get_articles_end=major_updates_in_range[-1],
                    )
                )
        output_dict[stock] = topic_list

    return output_dict


class GetNewsDevelopmentsAboutCompaniesInput(ToolArgs):
    stock_ids: List[StockID]
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function calls an internal API which provides all the news developments "
        "with articles between the start date and the end date that are relevant to the"
        " provided list of stocks, the output is a list of news developments. "
        "The default date_range is the last week. "
        "Never, ever pass an empty list of stocks to this function, you must only use this "
        "function when you have a specific list of stocks you want news for. If you have a general "
        "topic you want news about, use get_news_articles_for_topics. "
        "If the user asks about news sentiment, do NOT use this function, use the recommendation "
        "tool."
        "Never use this function to collect news for writing a commentary, instead use the "
        "get_commentary_inputs. "
        "Never use this tool together with write_commentary tool or get_commentary_inputs. "
        "Never, ever use get_default_text_data_for_stock as a substitute for this tool if the "
        "client says they want to look 'news' data for companies, you must use this tool "
        "even if multiple data sources are mentioned."
        " You should not pass a date_range containing dates after todays date into this function."
        " News documents can only be found for dates in the past up to the present, including todays date."
        " Sometimes you may be asked to read the news to prepare for, analyze or predict something "
        " about an upcoming event in the future,"
        " in those cases, when calling this function to get news, "
        " you MUST either use the default date_range or use a separate date_range from the past,"
        " the date_range used here must not be for the time period of the upcoming future event."
        " I repeat you will be FIRED if you try to find news from the future!!! YOU MUST NEVER DO THAT!!!"
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_all_news_developments_about_companies(
    args: GetNewsDevelopmentsAboutCompaniesInput, context: PlanRunContext
) -> List[StockNewsDevelopmentText]:
    if args.date_range:
        start_date, end_date = args.date_range.start_date, args.date_range.end_date
    else:
        start_date = None
        end_date = None
    topic_lookup = await _get_news_developments_helper(
        args.stock_ids, context.user_id, start_date, end_date
    )
    # if there are no news and there was no date range passed
    # then we extend the default to 1 month and try to get news again
    if start_date is None and end_date is None and len(topic_lookup.keys()) == 0:
        start_date = (get_now_utc() - datetime.timedelta(days=30)).date()
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = get_now_utc().date() + datetime.timedelta(days=1)
        topic_lookup = await _get_news_developments_helper(
            args.stock_ids, context.user_id, start_date, end_date
        )
    output: List[StockNewsDevelopmentText] = []
    for topic_list in topic_lookup.values():
        output.extend(topic_list)
    if len(output) == 0:
        stock_info = f"{len(args.stock_ids)} stocks"
        if len(args.stock_ids) < 10:
            stock_info = f"stocks: {args.stock_ids}"
        await tool_log(
            log=f"Did not get any news developments for {stock_info}"
            f" over the time period: {start_date=}, {end_date=}",
            context=context,
        )
    else:
        await tool_log(
            log=f"Found {len(output)} news developments for the provided stocks.", context=context
        )

    return output


class GetNewsArticlesForStockDevelopmentsInput(ToolArgs):
    developments_list: List[StockNewsDevelopmentText]


@tool(
    description=(
        "This function takes a list of news developments and returns a list of all the news"
        " development articles for those news developments. "
        "This function should be used if a client specifically mentions that they "
        "want to see individual news articles, rather than summarized developments. "
        "Do not convert the developments to articles unless it is very clear that the "
        "clients wants that level of detail."
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_news_articles_for_stock_developments(
    args: GetNewsArticlesForStockDevelopmentsInput, context: PlanRunContext
) -> List[StockNewsDevelopmentArticlesText]:
    sql = """
        SELECT news_id::VARCHAR, gbi_id, published_at
        FROM nlp_service.stock_news
        WHERE topic_id = ANY(%(topic_ids)s)
    """
    rows = get_psql().generic_read(
        sql, {"topic_ids": [topic.id for topic in args.developments_list]}
    )
    gbi_id_stock_map = {
        text.stock_id.gbi_id: text.stock_id for text in args.developments_list if text.stock_id
    }
    return [
        StockNewsDevelopmentArticlesText(
            id=row["news_id"],
            stock_id=gbi_id_stock_map.get(row["gbi_id"]),
            timestamp=row["published_at"],
        )
        for row in rows
    ]


THEME_RELEVANT_SYS_PROMPT = Prompt(name="THEME_RELEVANT_SYS_PROMPT", template="")
THEME_RELEVANT_MAIN_PROMPT = Prompt(
    name="THEME_RELEVANT_MAIN_PROMPT",
    template=(
        "You are a financial analyst checking to see if the following news article (headline + summary) "
        "is focused on a particular topic relevant to your investor clients ({topic}). "
        "The article should make clear, direct reference to the specific topic provided. "
        "If the topic has a direction (e.g. rising interest rates) you must also include articles that "
        "identify trends in the opposite direction (e.g. falling interest rates). Another requirement is"
        "that the news article must be serious reporting on a single specific event (this event should "
        "be in the title): you must say No to anything that looks like a listicle (e.g. 'Top 10...'), "
        "a topic overview, an opinion, or an advertisement. Finally, the event should be clearly important"
        "enough that it could potentially cause a significant movement in US stock prices. "
        "Answer Yes if the article is clearly focused on the provided topic and the article "
        "is serious reporting which refers to a specific, market-moving news development. "
        "You must answer No if any of the following are true: it is not specifically about "
        "the topic ({topic}), it is not a specific news development, or if it is not important enough "
        "to affect the markets. Do not write anything other than Yes or No. The topic is {topic}. "
        "Here is the article:\n ---\n{news}\n---\nYour answer:"
    ),
)


class GetNewsArticlesForTopicsInput(ToolArgs):
    topics: List[str]
    date_range: Optional[DateRange] = None
    max_num_articles_per_topic: Optional[int] = None


@tool(
    description=(
        "This function takes a list of topics and returns a "
        "list of news articles related to at least one of the given topics. (OR logic)"
        "If you need something which is about multiple topics at the same time (AND logic) "
        "Include it as a single topic joined by `and`"
        "If someone wants general information about a topic that is NOT a company/stock"
        "this is the best tool to call. "
        "If you do not set max_num_articles_per_topic, all are returned. "
        "This function must NEVER be used if you intend to filter stocks, the news articles do not "
        "contain information about which stocks they are relevant to. "
        "The default date range is the previous month. "
        "Never use this function to collect news for writing a commentary, instead use the "
        "get_commentary_inputs. "
        "Never use this tool together with write_commentary tool or get_commentary_inputs. "
        "Never use this tool with a stock/company/ticker in the topic, you MUST always use"
        "get_all_news_developments_about_companies to get news if the topic is a stock."
        "Again, if the client is interested in news about a particular stock, you must never, "
        "ever use this tool, you must get the news for that company."
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_news_articles_for_topics(
    args: GetNewsArticlesForTopicsInput, context: PlanRunContext
) -> List[NewsPoolArticleText]:
    # TODO: if start_date is very old, it will send many requests to GPT
    # start_date is optional, if not provided, use 30 days ago
    if args.date_range:
        start_date = args.date_range.start_date
        end_date = args.date_range.end_date
    else:
        start_date = (get_now_utc() - datetime.timedelta(days=30)).date()
        end_date = datetime.date.today() + datetime.timedelta(days=2)

    # prepare embedding
    llm = GPT(model=DEFAULT_EMBEDDING_MODEL)
    embeddings = [await llm.embed_text(topic) for topic in args.topics]

    # get news articles
    db = get_psql()
    news_articles: List[NewsPoolArticleText] = []
    for topic, embedding in zip(args.topics, embeddings):
        relevant_news: List[NewsPoolArticleText] = []
        for news_batch in _get_similar_news_to_embedding(
            db,
            start_date,
            end_date,
            embedding,
            batch_size=EMBEDDING_POOL_BATCH_SIZE,
        ):
            # check most relevant news with gpt
            gpt = GPT(model=DEFAULT_CHEAP_MODEL)
            tasks = []
            for _, news_text, _ in news_batch:
                tasks.append(
                    gpt.do_chat_w_sys_prompt(
                        main_prompt=THEME_RELEVANT_MAIN_PROMPT.format(topic=topic, news=news_text),
                        sys_prompt=THEME_RELEVANT_SYS_PROMPT.format(),
                    )
                )
            results = await gather_with_concurrency(tasks, n=EMBEDDING_POOL_BATCH_SIZE)

            successful = []
            for news, result in zip(news_batch, results):
                if result.lower().startswith("yes"):
                    successful.append(news)
            # if less than 10% of the batch is successful, stop
            if len(successful) / len(results) <= MIN_POOL_PERCENT_PER_BATCH:
                break

            relevant_news.extend(
                [NewsPoolArticleText(id=id, timestamp=timestamp) for id, _, timestamp in successful]
            )
            # if we have enough news, stop
            if len(relevant_news) >= MAX_NUM_RELEVANT_NEWS_PER_TOPIC:
                break

        await tool_log(
            log=f"Found {len(relevant_news)} news articles for topic: {topic}.",
            context=context,
            associated_data=relevant_news,
        )
        if args.max_num_articles_per_topic:
            relevant_news = relevant_news[: args.max_num_articles_per_topic]
        news_articles.extend(relevant_news)

    if len(news_articles) == 0:
        if args.date_range:
            raise EmptyOutputError(
                "Found no news articles for provided topic(s) within specified date range"
            )
        else:
            raise EmptyOutputError("Found no news articles for provided topic(s)")
    return news_articles


def _get_similar_news_to_embedding(
    db: Postgres,
    start_date: datetime.date,
    end_date: datetime.date,
    embedding: List[float],
    batch_size: int = 100,
) -> Generator[List[Tuple[str, str, datetime.datetime]], None, None]:
    sql = """
            SELECT
            news_id::TEXT, headline::TEXT, summary::TEXT, published_at,
            1 - (%s::VECTOR <=> embedding) AS similarity
            FROM nlp_service.news_pool
            WHERE published_at::DATE >= %s AND published_at::DATE <= %s
            ORDER BY similarity DESC, published_at DESC
            """

    with db.connection.cursor() as cursor:
        cursor.execute(sql, [str(embedding), start_date, end_date])
        rows = cursor.fetchmany(batch_size)
        while rows:
            yield [
                (
                    row["news_id"],  # type: ignore
                    f"Headline:{row['headline']}\nSummary:{row['summary']}",  # type: ignore
                    row["published_at"],  # type: ignore
                )
                for row in rows
            ]
            rows = cursor.fetchmany(batch_size)
