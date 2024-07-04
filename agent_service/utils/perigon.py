import json
import logging
import re
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, List, Optional
from urllib.parse import quote

import backoff
import requests
from gbi_common_py_utils.utils.ssm import get_param
from urllib3.exceptions import MaxRetryError, NewConnectionError

from agent_service.io_types.output import CitationOutput, CitationType

API_KEY_PARAM = "/perigon/api_key"
PERIGON_GPT_URL_TEMPLATE = (
    "https://api.goperigon.com/v1/answers/chatbot/threads/chat?apiKey={api_key}&content={query}"
)
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
                    if obj.strip():  # Skip any empty lines
                        try:
                            parsed_objects.append(json.loads(obj))
                        except json.JSONDecodeError as json_err:
                            print(f"JSON decode error: {json_err}")
                            print(f"Problematic content: {obj}")
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
                references.append(
                    CitationOutput(
                        citation_type=CitationType.LINK,
                        name=obj.get("title", ""),
                        link=obj.get("url"),
                        published_at=obj.get("pubDate"),
                        # perigon article id
                        article_id=obj.get("articleId"),
                    )
                )
        return PerigonNewsText(query=query, summary=summary, citations=references)
