from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from agent_service.GPT.constants import GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.utils.prompt_utils import Prompt

CLUSTER_PROMPT_COUNT_WEIGHT = 0.3
CLUSTER_USER_COUNT_WEIGHT = 0.3
# CLUSTER_DISTANCE_WEIGHT is important as a tie breaker
CLUSTER_DISTANCE_WEIGHT = 0.2
WITHIN_CLUSTER_WEIGHT = (
    1 - CLUSTER_PROMPT_COUNT_WEIGHT - CLUSTER_USER_COUNT_WEIGHT - CLUSTER_DISTANCE_WEIGHT
)


# This gets rid of specific mentions of stocks/universes/countries so that the embeddings focus more on the
# purpose of the query rather than the details. Should be done before embeddings are created

REMOVE_STOCKS_PROMPT = "You are a financial analyst that is helping to prepare data for training an automated finance chatbot. You are processing earlier requests to generalize the request by replacing certain details with their a placeholder. You will duplicate the input request verbatim, except that you will swap out a specific mention of any of the following types of things with a lower case string indicating their type: stocks (e.g. APPL), ETFs (e.g. QQQ), stock indexes (e.g. S&P 500), countries, regions, watchlists, and portfolios. Do not convert any other types. For example, 'show me how APPL did last week relevant to the rest of the S&P 500' would become 'show me how *stock* did last week relative to the rest of the *index*. Do not convert the type itself, only convert instances of the type. You must never, ever convert the word 'stock' or 'stocks' to '*stock*' or '*stocks*', for instance. For example, you would make no change to the following text: 'Show me stocks that are up 5%'. Only convert things that are clearly proper names. If there are no mention of any such items, you should return the input as is. Here is the input:\n{input}\nNow write your version with details removed:\n"  # noqa: E501


class QueryGeneralizer:
    def __init__(self) -> None:
        self.gpt = GPT(
            model=GPT4_O_MINI,
        )
        self.prompt = Prompt(REMOVE_STOCKS_PROMPT, "REMOVE_STOCKS_PROMPT")

    async def generalize(self, query: str) -> str:
        return await self.gpt.do_chat_w_sys_prompt(self.prompt.format(input=query), NO_PROMPT)


def rank_score(objects: List[Any]) -> List[float]:
    if len(objects) == 1:
        return [0.5]  # give a mid range score if there is only 1 user in a cluster
    else:
        return [(len(objects) - 1 - i) / (len(objects) - 1) for i in range(len(objects))]


def rank_by_similarity_with_centroid(embeddings: np.ndarray, centroid: np.ndarray) -> List[int]:
    cosine_similarities = cosine_similarity([centroid], embeddings)[0]
    return list(reversed(np.argsort(cosine_similarities)))


def get_scores_for_queries(
    query_ids: List[str], embedding_matrix: np.ndarray, user_lookup: Dict[str, str]
) -> Tuple[Dict[int, List[str]], Dict[int, float]]:
    # This function takes a list of query ids (Alfa user quries), a corresponding array of embeddings where each row
    # corresponds to embedding for the query, and a dictionary which maps query_ids to user_ids
    # it organizes the queries into near duplicate groups,
    # it returns a mapping of group nums into (near) duplicate query_id groups (queries in the duplicate groups are
    # ordered by centrality) and mapping of group_nums to scores between 0 and 1, close
    # to 1 means more important in terms of being a common, typical query

    clusterer = HDBSCAN(min_cluster_size=2, metric="cosine").fit(embedding_matrix)

    all_labels = set(clusterer.labels_)

    outliers_count = list(clusterer.labels_).count(-1)

    all_labels.remove(-1)

    cluster_groups: List[List[int]] = [[] for _ in range(len(all_labels) + outliers_count)]

    outliers_added = 0
    for i, label in enumerate(clusterer.labels_):
        if label == -1:
            cluster_groups[outliers_added + len(all_labels)].append(i)
            outliers_added += 1
        else:
            cluster_groups[label].append(i)

    # score each cluster by raw count

    cluster_raw_counts = {
        cluster_group_num: len(cluster_group)
        for cluster_group_num, cluster_group in enumerate(cluster_groups)
    }
    max_raw_count = max(cluster_raw_counts.values())
    query_count_score_lookup = {
        group_num: group_count / max_raw_count
        for group_num, group_count in cluster_raw_counts.items()
    }

    # score each cluster by number of users

    cluster_user_counts = {
        group_num: len(set([user_lookup[query_ids[query_idx]] for query_idx in group]))
        for group_num, group in enumerate(cluster_groups)
    }
    max_user_count = max(cluster_user_counts.values())
    user_count_score_lookup = {
        group_num: group_count / max_user_count
        for group_num, group_count in cluster_user_counts.items()
    }

    dup_groups: Dict[int, List[str]] = {}
    dup_group_scores: Dict[int, float] = {}

    # score each cluster by distance from space centroid

    centroid = np.sum(embedding_matrix, axis=0)

    cluster_centroids = {}

    for cluster_group_num, cluster_group in enumerate(cluster_groups):
        cluster_centroids[cluster_group_num] = np.sum(embedding_matrix[cluster_group], axis=0)

    cluster_similarity_order = rank_by_similarity_with_centroid(
        np.array(list(cluster_centroids.values())), centroid
    )

    cluster_nums = list(cluster_centroids)

    ordered_cluster_nums = []

    for order_num in cluster_similarity_order:
        ordered_cluster_nums.append(cluster_nums[order_num])

    distance_scores = rank_score(ordered_cluster_nums)

    distance_score_lookup = {
        cluster_num: score for cluster_num, score in zip(ordered_cluster_nums, distance_scores)
    }

    # within cluster scoring

    for cluster_group_num, cluster_group in enumerate(cluster_groups):
        user_groups = defaultdict(list)
        for query_idx in cluster_group:
            user_groups[user_lookup[query_ids[query_idx]]].append(query_idx)

        user_group_centroids = {}

        for user_id, user_group in user_groups.items():
            user_group_centroid = np.sum(embedding_matrix[user_group], axis=0)
            user_group_centroids[user_id] = user_group_centroid
            user_group_order = rank_by_similarity_with_centroid(
                embedding_matrix[user_group], user_group_centroid
            )
            new_user_group = []
            for order_num in user_group_order:
                new_user_group.append(user_group[order_num])
            user_groups[user_id] = (
                new_user_group  # reordered so the most central query is first in user group
            )

        ranked_user_idxs = rank_by_similarity_with_centroid(
            np.array(list(user_group_centroids.values())), cluster_centroids[cluster_group_num]
        )

        user_id_list = list(user_groups.keys())

        sorted_groups = []

        for idx in ranked_user_idxs:
            sorted_groups.append(user_groups[user_id_list[idx]])

        within_cluster_dist_scores = rank_score(sorted_groups)

        for group, within_cluster_dist_score in zip(sorted_groups, within_cluster_dist_scores):
            dup_groups[len(dup_groups)] = [query_ids[query_idx] for query_idx in group]
            score = (
                WITHIN_CLUSTER_WEIGHT * within_cluster_dist_score
                + CLUSTER_PROMPT_COUNT_WEIGHT * query_count_score_lookup[cluster_group_num]
                + CLUSTER_USER_COUNT_WEIGHT * user_count_score_lookup[cluster_group_num]
                + CLUSTER_DISTANCE_WEIGHT * distance_score_lookup[cluster_group_num]
            )
            dup_group_scores[len(dup_group_scores)] = score

    return dup_groups, dup_group_scores
