# imports for main function
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import backoff
import requests
from gbi_common_py_utils.utils.ssm import get_param
from urllib3.exceptions import MaxRetryError, NewConnectionError

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_types.output import CitationOutput, CitationType
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed

API_KEY_PARAM = "/perigon/api_key"

PERIGON_GPT_URL_TEMPLATE = (
    "https://api.goperigon.com/v1/answers/chatbot/threads/chat?apiKey={api_key}&content={query}"
)

PERIGON_ARTICLE_FETCH_URL_TEMPLATE = (
    "https://api.goperigon.com/v1/all?size=100&apiKey={api_key}&articleId={article_ids}"
)

PERIGON_SEMANTIC_URL_TEMPLATE = "https://api.goperigon.com/v1/answers/news/all?apiKey={api_key}"

PERIGON_RETRIABLE_EXCEPTION_TYPES = (
    ConnectionRefusedError,
    NewConnectionError,
    MaxRetryError,
    ConnectionError,
)

logger = logging.getLogger(__name__)


@dataclass
class PerigonNewsText:
    query: str
    summary: str
    citations: List[CitationOutput]


@dataclass
class PerigonArticle:
    citation: CitationOutput
    content: str


def convert_to_article_ids(
    news_list: List[Optional[PerigonNewsText]], extra_citations: List[CitationOutput]
) -> List[str]:
    article_ids = [
        citation.article_id
        for news_text in filter(None, news_list)
        for citation in news_text.citations
    ]

    article_ids.extend([citation.article_id for citation in extra_citations])

    return [article_id for article_id in article_ids if article_id]


class PerigonClient:
    def __init__(self) -> None:
        self.api_key = get_param(API_KEY_PARAM)

    @backoff.on_exception(
        backoff.expo,
        PERIGON_RETRIABLE_EXCEPTION_TYPES,
        max_time=120,  # 2 minutes in total
    )
    def _get_with_retries(self, url: str, query: str) -> Any:
        body = {"content": query}
        return requests.post(url, json=body)

    @backoff.on_exception(
        backoff.expo,
        PERIGON_RETRIABLE_EXCEPTION_TYPES,
        max_time=120,  # 2 minutes in total
    )
    def _get_semantic_with_retries(self, url: str, query: str, filters: Dict[str, str]) -> Any:
        body = {"prompt": query, "filter": filters, "size": 15}
        return requests.post(url, json=body)

    # list of news text returned from get_query, get a list of articles which were related to the news text
    def get_articles_from_news_text(self, article_ids: List[str]) -> List[PerigonArticle]:
        article_ids_string = ",".join(filter(None, article_ids))
        filled_url = PERIGON_ARTICLE_FETCH_URL_TEMPLATE.format(
            api_key=self.api_key, article_ids=article_ids_string
        )

        logger.info(f"Fetching articles from Perigon: {filled_url}")

        response = None
        retries = 0
        articles = []

        while response is None:
            try:
                response = requests.get(filled_url)
                raw_content = response.text
                json_content = json.loads(clean_to_json_if_needed(raw_content))
                articles = json_content.get("articles", [])
            except (ConnectionError, JSONDecodeError) as e:
                response = None
                retries += 1
                if retries < 5:
                    logger.warning(f"Failed to get articles from perigon due to {e}, retrying")
                    time.sleep(1)
                    retries += 1
                else:
                    logger.warning(f"Failed to get articles from perigon due to {e}, giving up")
                    return []

        result = [
            PerigonArticle(
                citation=CitationOutput(
                    citation_type=CitationType.LINK,
                    name=article["title"],
                    link=article["url"],
                    published_at=article["pubDate"],
                    article_id=article["articleId"],
                ),
                content=article["content"],
            )
            for article in filter(None, articles)
        ]

        return result

    def get_query_response(self, query: str) -> Optional[PerigonNewsText]:
        formatted_query = quote(query)
        filled_url = PERIGON_GPT_URL_TEMPLATE.format(api_key=self.api_key, query=query)

        logger.info(f"Making query to perigon: {filled_url}")

        response = None
        parsed_objects = None
        retries = 0
        while response is None:
            try:
                response = self._get_with_retries(filled_url, formatted_query)
                raw_content = response.text
                json_objects = raw_content.split("\n")
                parsed_objects = []
                for obj in json_objects:
                    if obj:
                        try:
                            parsed_objects.append(json.loads(clean_to_json_if_needed(obj)))
                        except JSONDecodeError as e:
                            logger.warning(f"Failed to parse json from perigon due to {e}")
            except (ConnectionError, JSONDecodeError) as e:
                response = None  # in case of json.loads() crash, force retry
                retries += 1
                if retries < 5:
                    logger.warning(f"Failed to get data from perigon due to {e}, retrying")
                    time.sleep(1)
                    retries += 1
                else:
                    logger.warning(f"Failed to get data from perigon due to {e}, giving up")
                    return None
        summary = ""
        references: List[CitationOutput] = []
        for obj in parsed_objects:
            if obj.get("type") == "RESPONSE_CHUNK":
                # removes [#] and ** from the summary
                summary = re.sub(r"\[\d+\]|\*\*", "", obj.get("content", ""))
            elif obj.get("object") == "NEWS_ARTICLE":
                try:
                    references.append(
                        CitationOutput(
                            citation_type=CitationType.LINK,
                            name=obj["title"],
                            link=obj["url"],
                            published_at=obj.get("pubDate"),
                            article_id=obj.get("articleId"),
                        )
                    )
                except KeyError as e:
                    logger.warning(f"Failed to get citation from perigon due to {e}")
        return PerigonNewsText(query=query, summary=summary, citations=references)

    def get_semantic_response(self, query: str, filters: Dict[str, str]) -> List[CitationOutput]:
        filled_url = PERIGON_SEMANTIC_URL_TEMPLATE.format(api_key=self.api_key)
        logger.info(f"Making semantic query to perigon to get citations: {filled_url}")

        response = None
        articles: List[Any] = []
        retries = 0
        while response is None:
            try:
                response = self._get_semantic_with_retries(filled_url, query, filters)
                raw_content = response.text
                json_content = json.loads(clean_to_json_if_needed(raw_content))
                articles = json_content.get("results", [])
            except (ConnectionError, JSONDecodeError) as e:
                response = None  # in case of json.loads() crash, force retry
                retries += 1
                if retries < 5:
                    logger.warning(f"Failed to get semantic data from perigon due to {e}, retrying")
                    time.sleep(1)
                    retries += 1
                else:
                    logger.warning(
                        f"Failed to get semantic data from perigon due to {e}, giving up"
                    )
                    return []

        result = [
            CitationOutput(
                citation_type=CitationType.LINK,
                name=article["data"]["title"],
                link=article["data"]["url"],
                published_at=article["data"]["pubDate"],
                article_id=article["data"]["articleId"],
            )
            for article in articles
            if article and article.get("data")
        ]

        return result


async def main() -> None:
    get_product_info_from_articles_str = (
        "Give me a list of {product} products from APPLE and exact specifications in a list like format which can be "
        "placed into a table"
    )

    get_product_info_from_articles_sys_str = (
        "Here are a list of articles to base your findings off of {articles}"
    )

    get_product_info_from_articles_sys = Prompt(
        name="GET_PRODUCT_COMPARE_SYS_PROMPT", template=get_product_info_from_articles_sys_str
    )

    get_product_info_from_articles = Prompt(
        name="GET_PRODUCT_COMPARE_MAIN_PROMPT", template=get_product_info_from_articles_str
    )

    pc = PerigonClient()
    semantic_result = pc.get_semantic_response(
        "What are specifications on Apple mobile phones?",
        {"language": "en", "sourceCountry": "us"},
    )

    result1 = pc.get_query_response("Give me the product specifications for Apple's mobile phones")
    result2 = pc.get_query_response("Give me a list of mobile phones from Apple")
    result3 = pc.get_query_response("Show me specifications of some Apple mobile phones")

    article_ids = convert_to_article_ids([result1, result2, result3], semantic_result)

    articles = pc.get_articles_from_news_text(article_ids)

    listed_article_content = [article.content for article in articles]

    llm = GPT(context=None, model=GPT4_O)
    result = await llm.do_chat_w_sys_prompt(
        main_prompt=get_product_info_from_articles.format(
            product="mobile phone",
        ),
        sys_prompt=get_product_info_from_articles_sys.format(articles=str(listed_article_content)),
    )

    print(len(listed_article_content))
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
