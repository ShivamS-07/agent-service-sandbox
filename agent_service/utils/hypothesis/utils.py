import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dateutil.parser import parse as date_parse
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import expit  # type: ignore

from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.hypothesis.constants import (
    EARNINGS_PERC_OF_NEWS_COUNT,
    HORIZON_DAY_LOOKUP,
    HORIZON_DELTA_LOOKUP,
    MIN_MAX_NEWS_COUNT,
    MIN_MAX_TOP_NEWS_COUNT,
    ONE_DAY,
    SUPPORT_GPT_WEIGHT,
    SUPPORT_GPT_WEIGHT_EARNINGS,
    SUPPORT_MULTIPLIER,
    SUPPORT_NEGATIVE_WEIGHT,
    SUPPORT_NEWS_COUNT_WEIGHT,
    SUPPORT_TOP_NEWS_COUNT_WEIGHT,
)
from agent_service.utils.hypothesis.types import (
    CompanyEarningsTopicInfo,
    CompanyNewsInfo,
    CompanyNewsTopicInfo,
    EarningsSummaryType,
    HypothesisEarningsTopicInfo,
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
        SELECT gbi_id, news_id::VARCHAR, topic_id::VARCHAR, headline, published_at, is_top_source
        FROM nlp_service.stock_news
        WHERE topic_id = ANY(%(topic_ids)s){time_filter}
    """
    rows = get_psql().generic_read(sql, params=params)

    return [CompanyNewsInfo(**row) for row in rows]


def get_max_news_count_pair_across_stocks(gbi_ids: List[int]) -> Tuple[int, int]:
    """
    Given a list of stocks, for ALL topics associated with each stock across ALL time, return
    the MAX number of news articles, and the MAX number of top sourced news articles across all
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
    max_news_count = max(
        (record["max_news_count"] for record in records if record["max_news_count"]),
        default=0,
    )
    max_top_source_count = max(
        (record["max_top_source_count"] for record in records if record["max_top_source_count"]),
        default=0,
    )

    return fix_max_pair((max_news_count, max_top_source_count))


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


def get_hypothesis_match_chart(
    news_hypothesis_topics: List[HypothesisNewsTopicInfo],
    earnings_hypothesis_topics: List[HypothesisEarningsTopicInfo],
    news_groups: List[List[CompanyNewsInfo]],
    max_count_pair: Tuple[int, int],
    chart_horizon: str = "3M",
    window_size: str = "1M",
    ref_time: Optional[datetime.datetime] = None,
) -> List[Tuple[datetime.date, float]]:
    """
    This function calculates the hypothesis match chart for the hypothesis report page
    First, we use the chart_horizon, window_size, and summary_start_time (the date which news summary
    for the relevant stock began) to identify the total range of dates we will need daily support values
    for. Then we iterate over the hypothesis-relevant topics and use their support score plus their news counts
    to get a support score for that topic, and split that score across the days based on the proportion of
    news articles for the day. Once we have daily support scores, we calculate a rolling sum of
    those values, weighted by recency, and pass them through logit function to get a range between -1 and 1
    We return a list of tuples corresponding to the date/averaged support score pairs, in order by date
    """
    ref_time = ref_time if ref_time is not None else get_now_utc()
    latest_end_of_day = closest_end_of_day(ref_time)
    # start dates 1M back from 3M ago
    first_date = (latest_end_of_day - HORIZON_DELTA_LOOKUP[chart_horizon]) - HORIZON_DELTA_LOOKUP[
        window_size
    ]
    dates = get_date_list(first_date, latest_end_of_day)

    date_lookup = {date: i for i, date in enumerate(dates)}
    # we split into pos and neg scores so we can properly calculate the desired bounds
    per_date_scores_pos, per_date_scores_neg = np.zeros((len(dates),)), np.zeros((len(dates),))

    if len(news_hypothesis_topics) != 0:
        for hypothesis_news_topic, news_group in zip(news_hypothesis_topics, news_groups):
            topic_match_score, date_counts = get_hypothesis_news_topic_match_score(
                hypothesis_news_topic, news_group, max_count_pair
            )
            for date, count in date_counts.items():
                if date in date_lookup:  # news might be from before the horizon window, ignore
                    score = topic_match_score * (count / len(news_group))
                    if score > 0:
                        per_date_scores_pos[date_lookup[date]] += score
                    else:
                        per_date_scores_neg[date_lookup[date]] += score

    if len(earnings_hypothesis_topics) != 0:
        for hypothesis_earnings_topic in earnings_hypothesis_topics:
            topic_match_score, date = get_hypothesis_earnings_topic_match_score(
                hypothesis_earnings_topic, max_count_pair
            )
            if date in date_lookup:
                if topic_match_score > 0:
                    per_date_scores_pos[date_lookup[date]] += topic_match_score
                else:
                    per_date_scores_neg[date_lookup[date]] += topic_match_score

    window_days = HORIZON_DAY_LOOKUP[window_size]
    recency_weights = get_recency_weights(window_days)
    weighed_data_pos = sliding_window_view(per_date_scores_pos, window_days) * recency_weights
    weighed_data_neg = sliding_window_view(per_date_scores_neg, window_days) * recency_weights

    # get bounds for calculation based on the idea that we don't want to vary between -1 and 1, but
    # instead between -bounds and bounds, where bounds is determined by taking the larger of the
    # value of all positive weights divided by the sum of all absolute weights and all negative
    # weights divided by the same sum. For example, if we had topic weights [1, 1, -0.5], the bound is
    # max((1 + 1)/(1 + 1 + 0.5), 0.5/(1 + 1 + 0.5)) = max(0.8, 0.2) = 0.8

    abs_sum = np.sum(weighed_data_pos + abs(weighed_data_neg), axis=1)

    # abs_sum could be zero
    bounds = np.nan_to_num(
        np.max(
            [
                np.sum(weighed_data_pos, axis=1) / abs_sum,
                np.sum(abs(weighed_data_neg), axis=1) / abs_sum,
            ],
            axis=0,
        )
    )

    # scale the raw sum of scores between -1 and 1, then clip to bounds
    daily_match_scores = np.clip(
        (expit(np.sum(weighed_data_pos + weighed_data_neg, axis=1) * SUPPORT_MULTIPLIER) - 0.5) * 2,
        a_max=bounds,
        a_min=-bounds,
    )
    return list(zip(dates[-len(daily_match_scores) :], daily_match_scores))


def get_hypothesis_earnings_topic_match_score(
    hypothesis_topic: HypothesisEarningsTopicInfo, max_count_pair: Tuple[int, int]
) -> Tuple[float, datetime.date]:
    perc_of_news_count = EARNINGS_PERC_OF_NEWS_COUNT
    support_weight = SUPPORT_GPT_WEIGHT_EARNINGS

    date = hypothesis_topic.summary_date.date()
    support_score: float = hypothesis_topic.get_latest_support(default=0)  # type:ignore
    polarity: Polarity = hypothesis_topic.get_latest_polarity(
        default=Polarity.neutral
    )  # type:ignore

    count = int(perc_of_news_count * max_count_pair[0])
    topic_match_score = get_topic_match_score(
        support_score, polarity, support_weight, max_count_pair, (count, count)
    )
    return topic_match_score, date


def get_hypothesis_topic_weights(
    hypothesis_topics: List[HypothesisNewsTopicInfo],
    news_groups: List[List[CompanyNewsInfo]],
    max_count_pair: Tuple[int, int],
    window_size: str = "1M",
    ref_time: Optional[datetime.datetime] = None,
) -> np.ndarray:  # 1d of floats same length as input topics
    """Calculates weights for individual topics so that the individual topic contributions
    correspond to their weight for support match score at ref_time with a window of window_size"""
    ref_time = ref_time if ref_time is not None else get_now_utc()
    latest_end_of_day = closest_end_of_day(ref_time)
    # start dates 1M back from 3M ago or when news summary started whichever is more recent
    start = latest_end_of_day - HORIZON_DELTA_LOOKUP[window_size]
    dates = get_date_list(start, latest_end_of_day)
    date_lookup = {date: i for i, date in enumerate(dates)}
    match_score_per_topic_date = np.zeros((len(hypothesis_topics), len(dates)))
    for i, (topic, news_group) in enumerate(zip(hypothesis_topics, news_groups)):
        match_score, date_counts = get_hypothesis_news_topic_match_score(
            topic, news_group, max_count_pair
        )
        for date, count in date_counts.items():
            if date in date_lookup:  # excludes dates from horizon
                match_score_per_topic_date[i, date_lookup[date]] = match_score * (
                    count / len(news_group)
                )

    window_days = HORIZON_DAY_LOOKUP[window_size]
    recency_weights = get_recency_weights(window_days)
    match_score_per_topic_date *= recency_weights
    return np.sum(match_score_per_topic_date, axis=1)  # Sum contribution of topic across all dates


def get_hypothesis_news_topic_match_score(
    hypothesis_topic: HypothesisNewsTopicInfo,
    news_group: List[CompanyNewsInfo],
    max_count_pair: Tuple[int, int],
) -> Tuple[float, Dict[datetime.date, int]]:
    # TODO: Calculate this using all supports, not just the last one
    support_score: float = hypothesis_topic.get_latest_support(default=0)  # type:ignore
    polarity: Polarity = hypothesis_topic.get_latest_polarity(
        default=Polarity.neutral
    )  # type:ignore
    date_counts: Dict[datetime.date, int] = defaultdict(int)
    top_news_count = 0
    for news_info in news_group:
        date_counts[news_info.published_at.date()] += 1
        if news_info.is_top_source:
            top_news_count += 1
    topic_match_score = get_topic_match_score(
        support_score,
        polarity,
        SUPPORT_GPT_WEIGHT,
        max_count_pair,
        (len(news_group), top_news_count),
    )
    return topic_match_score, date_counts


def get_topic_match_score(
    support_score: float,
    polarity: Polarity,
    support_weight: float,
    max_count_pair: Tuple[int, int],
    count_pair: Tuple[int, int],
) -> float:
    topic_match_score = support_score
    if topic_match_score < 0:
        multiplier = -1  # so that news weight counts negative
    else:
        multiplier = 1

    if polarity < 0:  # give more weight to negative topics, fight positive bias
        topic_match_score *= SUPPORT_NEGATIVE_WEIGHT
    topic_match_score *= support_weight

    max_news_count, max_top_news_count = max_count_pair
    news_group_count, top_news_group_count = count_pair
    topic_match_score += (
        get_clipped_log_count_weight(news_group_count, max_news_count)
        * SUPPORT_NEWS_COUNT_WEIGHT
        * multiplier
    )
    topic_match_score += (
        get_clipped_log_count_weight(top_news_group_count, max_top_news_count)
        * SUPPORT_TOP_NEWS_COUNT_WEIGHT
        * multiplier
    )
    return topic_match_score


def get_clipped_log_count_weight(count: int, max_count: int) -> float:
    return np.log2(min(count + 1, max_count + 1)) / np.log2(max_count + 1)
