import datetime
from collections import defaultdict
from typing import Dict, Generator, List, Optional

from agent_service.external.grpc_utils import timestamp_to_date
from agent_service.external.nlp_svc_client import get_multi_companies_news_topics
from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL, DEFAULT_EMBEDDING_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_types.dates import DateRange
from agent_service.io_types.stock import StockID
from agent_service.io_types.text import (
    NewsPoolArticleText,
    StockNewsDevelopmentArticlesText,
    StockNewsDevelopmentText,
)
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.postgres import Postgres, get_psql
from agent_service.utils.prompt_utils import Prompt

EMBEDDING_POOL_BATCH_SIZE = 100
MIN_POOL_PERCENT_PER_BATCH = 0.1
MAX_NUM_RELEVANT_NEWS_PER_TOPIC = 200


async def _get_news_developments_helper(
    stock_ids: List[StockID],
    user_id: str,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
) -> Dict[StockID, List[StockNewsDevelopmentText]]:
    response = await get_multi_companies_news_topics(
        user_id=user_id, gbi_ids=[stock.gbi_id for stock in stock_ids]
    )
    # Response now has a list of topics. Build an association dict to ensure correct ordering.
    stock_to_topics_map: Dict[StockID, List] = defaultdict(list)
    gbi_id_to_topics_map: Dict[int, StockID] = {stock.gbi_id: stock for stock in stock_ids}
    for topic in response.topics:
        stock = gbi_id_to_topics_map[topic.gbi_id]
        stock_to_topics_map[stock].append(topic)

    if not start_date:
        start_date = (get_now_utc() - datetime.timedelta(days=7)).date()
    if not end_date:
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = get_now_utc().date() + datetime.timedelta(days=1)

    output_dict: Dict[StockID, List[StockNewsDevelopmentText]] = {}
    for stock in stock_ids:
        topics = stock_to_topics_map[stock]
        topic_list = []
        for topic in topics:
            topic_date = timestamp_to_date(topic.last_article_date)
            if topic_date is None or topic_date < start_date or topic_date > end_date:
                # Filter topics not in the time window
                continue
            # Only return ID's
            topic_list.append(StockNewsDevelopmentText(id=topic.topic_id.id, stock_id=stock))
        output_dict[stock] = topic_list

    return output_dict


class GetNewsDevelopmentsAboutCompaniesInput(ToolArgs):
    stock_ids: List[StockID]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None


@tool(
    description=(
        "This function calls an internal API which provides all the news developments "
        "with articles between the start date and the end date that are relevant to the"
        " provided list of stocks, the output is a list of news developments. "
        "If end_date is left out, "
        "the current date is used. If start_date is left out, 1 week ago is used."
        "Never, ever pass an empty list of stocks to this function, you must only use this "
        "function when you have a specific list of stocks you want news for. If you have a general "
        "topic you want news about, use get_news_articles_for_topics. "
        "If the user asks about news sentiment, do NOT use this function, use the recommendation "
        "tool."
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_all_news_developments_about_companies(
    args: GetNewsDevelopmentsAboutCompaniesInput, context: PlanRunContext
) -> List[StockNewsDevelopmentText]:
    start_date = args.start_date
    end_date = args.end_date
    if args.date_range:
        start_date, end_date = args.date_range.start_date, args.date_range.end_date
    topic_lookup = await _get_news_developments_helper(
        args.stock_ids, context.user_id, start_date, end_date
    )
    output: List[StockNewsDevelopmentText] = []
    for topic_list in topic_lookup.values():
        output.extend(topic_list)
    if len(output) == 0:
        raise Exception("Did not get any news developments for these stocks over the time period")

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
        SELECT news_id::VARCHAR, gbi_id
        FROM nlp_service.stock_news
        WHERE topic_id = ANY(%(topic_ids)s)
    """
    rows = get_psql().generic_read(
        sql, {"topic_ids": [topic.id for topic in args.developments_list]}
    )
    gbi_id_stock_map = {
        text.stock_id.gbi_id: text.stock_id for text in args.developments_list if text.stock_id
    }
    if not rows:
        raise Exception("No articles for these news developments over the specified time period")
    return [
        StockNewsDevelopmentArticlesText(
            id=row["news_id"], stock_id=gbi_id_stock_map.get(row["gbi_id"])
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
    start_date: Optional[datetime.date] = None
    date_range: Optional[DateRange] = None
    max_num_articles_per_topic: Optional[int] = None


@tool(
    description=(
        "This function takes a list of topics and returns a "
        "list of news articles related to the given topics. "
        "If someone wants general information about a topic and there is no existing themes "
        "This is the best tool to call. "
        "This function must NEVER be used if you intend to filter stocks, the news articles do not "
        "contain information about which stocks they are relevant to."
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_news_articles_for_topics(
    args: GetNewsArticlesForTopicsInput, context: PlanRunContext
) -> List[NewsPoolArticleText]:
    # TODO: if start_date is very old, it will send many requests to GPT
    # start_date is optional, if not provided, use 30 days ago
    if not args.start_date:
        if args.date_range:
            start_date = args.date_range.start_date
        else:
            start_date = (get_now_utc() - datetime.timedelta(days=30)).date()
    else:
        start_date = args.start_date

    # prepare embedding
    llm = GPT(model=DEFAULT_EMBEDDING_MODEL)
    embeddings = [await llm.embed_text(topic) for topic in args.topics]

    # get news articles
    db = get_psql()
    news: List[NewsPoolArticleText] = []
    for topic, embedding in zip(args.topics, embeddings):
        relevant_news: List[NewsPoolArticleText] = []
        for news_ids in _get_similar_news_to_embedding(
            db, start_date, embedding, batch_size=EMBEDDING_POOL_BATCH_SIZE
        ):
            # check most relevant news with gpt
            gpt = GPT(model=DEFAULT_CHEAP_MODEL)
            tasks = []
            for news_text in news_ids.values():
                tasks.append(
                    gpt.do_chat_w_sys_prompt(
                        main_prompt=THEME_RELEVANT_MAIN_PROMPT.format(topic=topic, news=news_text),
                        sys_prompt=THEME_RELEVANT_SYS_PROMPT.format(),
                    )
                )
            results = await gather_with_concurrency(tasks, n=EMBEDDING_POOL_BATCH_SIZE)

            successful = []
            for news_id, result in zip(news_ids.keys(), results):
                if result.lower().startswith("yes"):
                    successful.append(news_id)
            # if less than 10% of the batch is successful, stop
            if len(successful) / len(results) <= MIN_POOL_PERCENT_PER_BATCH:
                break

            relevant_news.extend([NewsPoolArticleText(id=id) for id in successful])
            # if we have enough news, stop
            if len(relevant_news) >= MAX_NUM_RELEVANT_NEWS_PER_TOPIC:
                break

        await tool_log(
            log=f"Found {len(relevant_news)} news articles for topic: {topic}.",
            context=context,
        )
        if args.max_num_articles_per_topic:
            relevant_news = relevant_news[: args.max_num_articles_per_topic]
        news.extend(relevant_news)

    if len(news) == 0:
        raise Exception("Found no news articles for provided topic(s)")
    return news


def _get_similar_news_to_embedding(
    db: Postgres, start_date: datetime.date, embedding: List[float], batch_size: int = 100
) -> Generator[Dict[str, str], None, None]:
    sql = """
            SELECT
            news_id::TEXT, headline::TEXT, summary::TEXT,
            1 - (%s::VECTOR <=> embedding) AS similarity
            FROM nlp_service.news_pool
            WHERE published_at >= %s
            ORDER BY similarity DESC, published_at DESC
            """

    with db.connection.cursor() as cursor:
        cursor.execute(sql, [str(embedding), start_date])
        rows = cursor.fetchmany(batch_size)
        while rows:
            yield {
                row["news_id"]: f"Headline:{row['headline']}\nSummary:{row['summary']}"
                for row in rows
            }
            rows = cursor.fetchmany(batch_size)
