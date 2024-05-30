import datetime
from collections import defaultdict
from typing import Dict, Generator, List, Optional

from agent_service.external.grpc_utils import timestamp_to_datetime
from agent_service.external.nlp_svc_client import get_multi_companies_news_topics
from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL, DEFAULT_EMBEDDING_MODEL
from agent_service.GPT.requests import GPT
from agent_service.io_types.text import (
    NewsPoolText,
    StockAlignedTextGroups,
    StockNewsDevelopmentArticlesText,
    StockNewsDevelopmentText,
    TextGroup,
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
    gbi_ids: List[int],
    user_id: str,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
) -> Dict[int, List[StockNewsDevelopmentText]]:
    response = await get_multi_companies_news_topics(user_id=user_id, gbi_ids=gbi_ids)
    # Response now has a list of topics. Build an association dict to ensure correct ordering.
    stock_to_topics_map: Dict[int, List] = defaultdict(list)
    for topic in response.topics:
        stock_to_topics_map[topic.gbi_id].append(topic)

    if not start_date:
        start_date = (get_now_utc() - datetime.timedelta(days=7)).date()
    if not end_date:
        # Add an extra day to be sure we don't miss anything with timezone weirdness
        end_date = get_now_utc().date() + datetime.timedelta(days=1)

    output_dict: Dict[int, List[StockNewsDevelopmentText]] = {}
    for gbi_id in gbi_ids:
        topics = stock_to_topics_map[gbi_id]
        topic_list = []
        for topic in topics:
            topic_date = timestamp_to_datetime(topic.last_article_date).date()
            if topic_date < start_date or topic_date > end_date:
                # Filter topics not in the time window
                continue
            # Only return ID's
            topic_list.append(StockNewsDevelopmentText(id=topic.topic_id.id))
        output_dict[gbi_id] = topic_list

    return output_dict


class GetNewsDevelopmentsAboutCompaniesInput(ToolArgs):
    stock_ids: List[int]
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None


@tool(
    description=(
        "This function calls an internal API which provides all the news developments "
        "with articles between the start date and the end date that are relevant to the"
        " provided list of stocks, the output is a list of news developments. "
        "Unlike get_stock_aligned_news_developments, all developments are returned in a single list"
        "there is no segregation by company, so it is appropriately used when you are filtering "
        "and or summarizing all news about one or more stocks into a single summary. "
        "This function is not appropriate for use in filtering of the input stocks, "
        "or other applications where you need to do per stock analysis or per stock generation of "
        "text since all the news is included together and it is not possible to do anything else "
        "at an individual stock level"
        "An example of the kind of query you would use this function for: "
        " `Summarize all the news about GPUs for Nvida and Intel over the last 3 weeks. "
        "If you want to filter news articles by topic, you should choose this function. "
        "If end_date is left out, "
        "the current date is used. If start_date is left out, 1 week ago is used."
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_all_news_developments_about_companies(
    args: GetNewsDevelopmentsAboutCompaniesInput, context: PlanRunContext
) -> List[StockNewsDevelopmentText]:
    topic_lookup = await _get_news_developments_helper(
        args.stock_ids, context.user_id, args.start_date, args.end_date
    )
    output: List[StockNewsDevelopmentText] = []
    for topic_list in topic_lookup.values():
        output.extend(topic_list)
    return output


@tool(
    description=(
        "This function calls an internal API which provides all the news developments "
        "with articles between the start date and the end date, arranged "
        " according to stock, the output is a list of StockAlignedTextGroups with a "
        "mapping from stocks to lists of news articles associated with the stock "
        "This function should be used when you plan to pass this data to an LLM-based aligned function"
        " to filter stocks. Use get_all_news_developments_about_companies if you simply want"
        " to summarize all the news. An example of the kind of query you would use this for: "
        "`I want a list of airline stocks that have faced major customer service issues in the last month. "
        "Again, if you want to filter stocks by topic, you should choose this function."
        "If end_date is left out, the current date is used. If start_date is left out, 1 week ago is used"
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_stock_aligned_news_developments(
    args: GetNewsDevelopmentsAboutCompaniesInput, context: PlanRunContext
) -> StockAlignedTextGroups:
    topic_lookup = await _get_news_developments_helper(
        args.stock_ids, context.user_id, args.start_date, args.end_date
    )
    output: Dict[int, TextGroup] = {}
    for stock_id, topic_list in topic_lookup.items():
        output[stock_id] = TextGroup(val=topic_list)  # type: ignore
    return StockAlignedTextGroups(val=output)


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
        SELECT news_id::VARCHAR
        FROM nlp_service.stock_news
        WHERE topic_id = ANY(%(topic_ids)s)
    """
    rows = get_psql().generic_read(
        sql, {"topic_ids": [topic.id for topic in args.developments_list]}
    )
    return [StockNewsDevelopmentArticlesText(id=row["news_id"]) for row in rows]


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


@tool(
    description=(
        "This function takes a list of topics and returns a "
        "list of news articles related to the given topics. The output is a list of NewsPoolText objects, "
        "each containing a news article. "
    ),
    category=ToolCategory.NEWS,
    tool_registry=ToolRegistry,
)
async def get_news_articles_for_topics(
    args: GetNewsArticlesForTopicsInput, context: PlanRunContext
) -> List[List[NewsPoolText]]:
    # TODO: if start_date is very old, it will send many reuquests to GPT
    # start_date is optional, if not provided, use 30 days ago
    start_date = args.start_date
    if not start_date:
        start_date = (get_now_utc() - datetime.timedelta(days=30)).date()

    # prepare embedding
    llm = GPT(model=DEFAULT_EMBEDDING_MODEL)
    embeddings = [await llm.embed_text(topic) for topic in args.topics]

    # get news articles
    db = get_psql()
    news: List[List[NewsPoolText]] = []
    for topic, embedding in zip(args.topics, embeddings):
        relevant_news: List[NewsPoolText] = []
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

            relevant_news.extend([NewsPoolText(id=id) for id in successful])
            # if we have enough news, stop
            if len(relevant_news) >= MAX_NUM_RELEVANT_NEWS_PER_TOPIC:
                break

        await tool_log(
            log=f"Found {len(relevant_news)} news articles for topic {topic}.",
            context=context,
        )
        news.append(relevant_news)

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
