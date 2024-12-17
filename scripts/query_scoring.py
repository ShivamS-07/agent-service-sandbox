import asyncio
import datetime
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from agent_service.GPT.requests import GPT
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.postgres import Postgres
from agent_service.utils.query_scoring import QueryGeneralizer, get_scores_for_queries

db = Postgres(environment="ALPHA")


def get_first_messages(start: datetime.datetime, end: datetime.datetime) -> List[Dict[str, Any]]:
    # check the create and update times for the plan id
    # we do this because currently the create date is generated on db server
    # at row insert but we supply a last_updated datetime in code when updating status.
    # return any plan runs within the time period
    sql = """
        SELECT DISTINCT ON (agent_id) message_id::TEXT, agent_id::TEXT, message_author::TEXT, message, message_time
        FROM agent.chat_messages
        WHERE message_time > %(start)s
              AND message_time < %(end)s
              AND is_user_message
        ORDER BY agent_id, message_time ASC
    """

    params = {"start": start, "end": end}
    rows = db.generic_read(sql, params=params)
    return rows


CHUNK_SIZE = 40


def filter_out_templates(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sql = """
        SELECT prompt
        FROM agent.prompt_templates
    """
    rows = db.generic_read(sql)

    template_chunks = set()

    for row in rows:
        prompt = row["prompt"]
        for i in range(len(prompt) - CHUNK_SIZE):
            template_chunks.add(prompt[i : i + CHUNK_SIZE])

    filtered_messages = []

    for message in messages:
        found = False
        prompt = message["message"]
        for i in range(len(prompt) - CHUNK_SIZE):
            if prompt[i : i + CHUNK_SIZE] in template_chunks:
                found = True
                break
        if not found:
            filtered_messages.append(message)
    return filtered_messages


def filter_to_external_users(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    sql = """
        SELECT id::TEXT, email
        FROM user_service.users
        WHERE id = ANY(%(user_ids)s)
    """

    users = list(set([message["message_author"] for message in messages]))

    params = {"user_ids": users}
    rows = db.generic_read(sql, params=params)
    email_lookup = {}
    for row in rows:
        email_lookup[row["id"]] = row["email"]
    wanted_ids = set([row["id"] for row in rows if "boosted" not in row["email"]])
    return [
        message for message in messages if message["message_author"] in wanted_ids
    ], email_lookup


async def main() -> None:
    # now = get_now_utc()
    now = datetime.datetime(year=2024, month=11, day=28, hour=12, minute=0, second=0)
    then = now - datetime.timedelta(days=60)
    messages = get_first_messages(then, now)
    # messages = messages[:50]
    messages, email_lookup = filter_to_external_users(messages)
    messages = filter_out_templates(messages)

    # all this saving is just for development convenience

    if os.path.exists("/tmp/query_cluster/message_generalize_cache.pkl"):
        with open("/tmp/query_cluster/message_generalize_cache.pkl", "rb") as f:
            message_generalized_lookup = pickle.load(f)
    else:
        message_generalized_lookup = {}

    query_generalizer = QueryGeneralizer()

    tasks = []

    for message in messages:
        if message["message_id"] not in message_generalized_lookup:
            tasks.append(query_generalizer.generalize(message["message"]))
        else:
            tasks.append(identity(message_generalized_lookup[message["message_id"]]))

    results = await gather_with_concurrency(tasks=tasks, n=100)

    for message, result in zip(messages, results):
        message_generalized_lookup[message["message_id"]] = result

    with open("/tmp/query_cluster/message_generalize_cache.pkl", "wb") as fout:
        pickle.dump(message_generalized_lookup, fout)

    if os.path.exists("/tmp/query_cluster/embedding_cache.pkl"):
        with open("/tmp/query_cluster/embedding_cache.pkl", "rb") as f:
            message_id_list, old_embedding_matrix = pickle.load(f)
    else:
        message_id_list = []
        old_embedding_matrix = None

    message_set = set(message_id_list)
    tasks = []
    for message in messages:
        if message["message_id"] not in message_set:
            tasks.append(
                GPT().embed_text(message_generalized_lookup[message["message_id"]][:10000])
            )
            message_id_list.append(message["message_id"])

    if tasks:
        embeddings = await gather_with_concurrency(tasks)
        new_embedding_matrix = np.vstack(embeddings)

        if old_embedding_matrix is not None:
            embedding_matrix = np.vstack([old_embedding_matrix, new_embedding_matrix])
        else:
            embedding_matrix = new_embedding_matrix
        save_matrix = True
    else:
        embedding_matrix = old_embedding_matrix
        save_matrix = False

    if save_matrix:
        with open("/tmp/query_cluster/embedding_cache.pkl", "wb") as fout:
            pickle.dump((message_id_list, embedding_matrix), fout)

    message_lookup = {message["message_id"]: message for message in messages}
    new_message_id_list = []
    bool_list = []
    for message_id in message_id_list:
        if message_id in message_lookup:
            new_message_id_list.append(message_id)
            bool_list.append(True)
        else:
            bool_list.append(False)

    embedding_matrix = embedding_matrix[bool_list]
    message_id_list = new_message_id_list

    user_lookup = {
        message_id: message_lookup[message_id]["message_author"] for message_id in message_id_list
    }
    query_lookup = {
        message_id: message_lookup[message_id]["message"] for message_id in message_id_list
    }

    (
        dup_clusters,
        dup_scores,
    ) = get_scores_for_queries(message_id_list, embedding_matrix, user_lookup)

    count = 0
    for score, cluster_num in sorted(
        [(score, cluster_num) for cluster_num, score in dup_scores.items()], reverse=True
    ):
        print(count)
        print(score)
        print(email_lookup[user_lookup[dup_clusters[cluster_num][0]]])
        for message_idx in dup_clusters[cluster_num]:
            print("----")
            print(query_lookup[message_idx])

        print("*****")
        if count == 100:
            break
        count += 1


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
