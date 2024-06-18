import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dateutil.parser import parse as date_parse

from agent_service.utils.hypothesis.constants import (
    MIN_MAX_NEWS_COUNT,
    MIN_MAX_TOP_NEWS_COUNT,
    ONE_DAY,
)
from agent_service.utils.hypothesis.types import (
    CompanyEarningsTopicInfo,
    CompanyNewsInfo,
    CompanyNewsTopicInfo,
    EarningsSummaryType,
    HypothesisNewsTopicInfo,
    NewsImpact,
    Polarity,
)
from agent_service.utils.postgres import get_psql


def get_news_topics(topic_ids: List[str]) -> List[CompanyNewsTopicInfo]:
    if isinstance(topic_ids, list) and len(topic_ids) == 0:
        return []

    sql = """
        SELECT gbi_id, topic_id::VARCHAR, topic_label, topic_descriptions, topic_polarities,
            topic_impacts
        FROM nlp_service.stock_news_topics
        WHERE topic_id = ANY(%(topic_ids)s)
        ORDER BY created_at DESC
    """
    records = get_psql().generic_read(sql, params={"topic_ids": topic_ids})
    outputs = []
    for record in records:
        outputs.append(
            CompanyNewsTopicInfo(
                topic_id=record["topic_id"],
                gbi_id=record["gbi_id"],
                topic_label=record["topic_label"],
                topic_descriptions=[
                    (tup[0], datetime.datetime.fromisoformat(tup[1]))
                    for tup in record["topic_descriptions"]
                ],
                topic_polarities=[
                    (Polarity(tup[0]), datetime.datetime.fromisoformat(tup[1]))
                    for tup in record["topic_polarities"]
                ],
                topic_impacts=(
                    [
                        (NewsImpact(tup[0]), datetime.datetime.fromisoformat(tup[1]))
                        for tup in record["topic_impacts"]
                    ]
                    if record["topic_impacts"] is not None
                    else []
                ),
            )
        )

    return outputs


def get_news_from_topics(
    topic_ids: List[str],
    min_time: Optional[datetime.datetime] = None,
    max_time: Optional[datetime.datetime] = None,
) -> List[CompanyNewsInfo]:
    params: Dict[str, Any] = {"topic_ids": topic_ids}

    time_filter = ""
    if min_time:
        params["min_time"] = min_time
        time_filter = "AND published_at >= %(min_time)s"

    if max_time:
        params["max_time"] = max_time
        time_filter = "AND published_at <= %(max_time)s"

    sql = f"""
        SELECT news_id::VARCHAR, topic_id::VARCHAR, headline, published_at, is_top_source
        FROM nlp_service.stock_news
        WHERE topic_id = ANY(%(topic_ids)s){time_filter}
    """
    rows = get_psql().generic_read(sql, params=params)

    return [CompanyNewsInfo(**row) for row in rows]


def get_max_news_count_pair_for_stocks(gbi_ids: List[int]) -> Dict[int, Tuple[int, int]]:
    """
    Given a list of stocks, for ALL topics associated with each stock across ALL time, return
    the MAX number of news articles, and the MAX number of top sourced news articles for each
    stock

    Returns:
        Dict[int, Tuple[int]]: A dictionary of gbi_id to a tuple of (max_news_count, max_top_source_count)
    """
    sql = """
        WITH max_news_count AS (
            SELECT DISTINCT ON (sn.gbi_id) sn.gbi_id, COUNT(sn.news_id) AS max_news_count
            FROM nlp_service.stock_news sn
            WHERE gbi_id = ANY(%(gbi_ids)s) AND sn.topic_id NOTNULL
            GROUP BY sn.gbi_id, sn.topic_id
            ORDER BY sn.gbi_id, max_news_count DESC
        ), max_top_source_count AS (
            SELECT DISTINCT ON (sn.gbi_id) sn.gbi_id, SUM(sn.is_top_source::INT) AS max_top_source_count
            FROM nlp_service.stock_news sn
            WHERE gbi_id = ANY(%(gbi_ids)s) AND sn.topic_id NOTNULL
            GROUP BY sn.gbi_id, sn.topic_id
            ORDER BY sn.gbi_id, max_top_source_count DESC
        )
        SELECT mnc.gbi_id, mnc.max_news_count, mtsc.max_top_source_count
        FROM max_news_count mnc
        LEFT JOIN max_top_source_count mtsc ON mnc.gbi_id = mtsc.gbi_id;
    """
    records = get_psql().generic_read(sql, params={"gbi_ids": gbi_ids})
    return {
        record["gbi_id"]: (record["max_news_count"], record["max_top_source_count"])
        for record in records
    }


def fix_max_pair(max_count_pair: Tuple[Optional[int], Optional[int]]) -> Tuple[int, int]:
    # This stops random small news from being considered hugely important because is the only news
    # the company has ever seen
    res1 = (
        MIN_MAX_NEWS_COUNT if not max_count_pair[0] else max(max_count_pair[0], MIN_MAX_NEWS_COUNT)
    )
    res2 = (
        MIN_MAX_TOP_NEWS_COUNT
        if not max_count_pair[1]
        else max(max_count_pair[1], MIN_MAX_TOP_NEWS_COUNT)
    )
    return (res1, res2)


def news_sort(news_infos: List[CompanyNewsInfo]) -> List[CompanyNewsInfo]:
    news_infos.sort(key=lambda x: (x.is_top_source, x.published_at, x.headline), reverse=True)
    return news_infos


def convert_to_news_groups(
    topics: Union[List[CompanyNewsTopicInfo], List[HypothesisNewsTopicInfo]],
    news: List[CompanyNewsInfo],
) -> List[List[CompanyNewsInfo]]:
    news_groups: List[List[CompanyNewsInfo]] = [[] for _ in topics]
    topic_lookup = {topic.topic_id: i for i, topic in enumerate(topics)}  # type: ignore
    for news_info in news:
        if news_info.topic_id in topic_lookup:
            news_groups[topic_lookup[news_info.topic_id]].append(news_info)
    for news_group in news_groups:
        news_sort(news_group)
    return news_groups


def date_to_utc_time(date: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(date, datetime.time.max).replace(tzinfo=datetime.timezone.utc)


def closest_end_of_day(ref_time: datetime.datetime) -> datetime.datetime:
    if ref_time.hour < 12:
        return date_to_utc_time((ref_time - ONE_DAY).date())
    else:
        return date_to_utc_time(ref_time.date())


def get_date_list(
    start_time: datetime.datetime, end_time: datetime.datetime
) -> List[datetime.date]:
    curr_time = end_time
    dates = []
    while curr_time > start_time:
        dates.append(curr_time.date())
        curr_time -= ONE_DAY
    return dates[::-1]


def get_recency_weights(window_days: int) -> np.ndarray:
    return np.array([i / window_days for i in range(1, window_days + 1)])


def get_earnings_topics(summary_ids: List[str]) -> List[CompanyEarningsTopicInfo]:
    sql = """
        SELECT summary_id::TEXT, gbi_id, summary, sources
        FROM nlp_service.earnings_call_summaries
        WHERE summary_id = ANY(%(summary_ids)s)
    """
    rows = get_psql().generic_read(sql, params={"summary_ids": summary_ids})

    outputs = []
    for row in rows:
        summary_id = row["summary_id"]
        gbi_id = row["gbi_id"]
        earnings_date = date_parse(row["sources"][0]["publishing_time"])
        summary: Dict[str, Any] = row["summary"]

        for idx, point in enumerate(summary.get("Remarks", [])):
            outputs.append(
                CompanyEarningsTopicInfo(
                    topic_id=summary_id,
                    gbi_id=gbi_id,
                    topic_label=point["header"],
                    topic_descriptions=[(point["detail"], earnings_date)],
                    topic_polarities=[
                        (Polarity.from_sentiment_str(point["sentiment"]), earnings_date)
                    ],
                    topic_impacts=[(NewsImpact.medium, earnings_date)],
                    summary_index=idx,
                    summary_type=EarningsSummaryType.REMARKS,
                    summary_date=earnings_date,
                )
            )

        for idx, point in enumerate(summary.get("Questions", [])):
            outputs.append(
                CompanyEarningsTopicInfo(
                    topic_id=summary_id,
                    gbi_id=gbi_id,
                    topic_label=point["header"],
                    topic_descriptions=[(point["detail"], earnings_date)],
                    topic_polarities=[
                        (Polarity.from_sentiment_str(point["sentiment"]), earnings_date)
                    ],
                    topic_impacts=[(NewsImpact.medium, earnings_date)],
                    summary_index=idx,
                    summary_type=EarningsSummaryType.QUESTIONS,
                    summary_date=earnings_date,
                )
            )

    return outputs
