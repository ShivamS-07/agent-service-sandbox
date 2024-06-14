import asyncio
import datetime
import logging
from typing import List, Optional, Tuple, Union

from agent_service.GPT.constants import TEXT_3_LARGE
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.hypothesis.constants import (
    IRRELEVANT_TOPICS_THRESHOLD,
    MAX_BATCHES,
    NEWS_TOPICS_BATCH_SIZE,
    NUM_TOPIC_WORKERS,
    NUM_TOPICS_UB,
    TOTAL_RELEVANT_TOPICS_THRESHOLD,
)
from agent_service.utils.hypothesis.hypothesis_ai import HypothesisAI
from agent_service.utils.hypothesis.types import (
    CompanyNewsTopicInfo,
    HypothesisEarningsTopicInfo,
    HypothesisInfo,
    HypothesisNewsTopicInfo,
)
from agent_service.utils.hypothesis.utils import get_news_topics
from agent_service.utils.postgres import get_psql

logger = logging.getLogger(__name__)


class HypothesisPipeline:
    def __init__(
        self, gbi_id: int, hypothesis_text: str, ref_time: Optional[datetime.datetime] = None
    ) -> None:
        self.ref_time = ref_time if ref_time else get_now_utc()

        self.pg = get_psql()
        self.ch = Clickhouse()

        self.hypothesis = self.create_hypothesis_info(gbi_id, hypothesis_text)

        self.llm = HypothesisAI(self.hypothesis, ref_time=self.ref_time)

    def create_hypothesis_info(self, gbi_id: int, hypothesis_text: str) -> HypothesisInfo:
        stock_metadata = self.pg.get_sec_metadata_from_gbi([gbi_id]).get(gbi_id)
        if stock_metadata is None:
            raise ValueError("Could not find stock metadata.")
        company_name = stock_metadata.company_name

        description = self.pg.get_short_company_description(gbi_id)[0]
        company_description = description if description else "No description"

        return HypothesisInfo(
            hypothesis_text=hypothesis_text,
            gbi_id=gbi_id,
            company_name=company_name,
            company_description=company_description,
        )

    async def initial_hypothesis_processing(self) -> None:
        logger.info("Creating hypothesis breakdown...")
        await self.llm.get_hypothesis_breakdown()
        logger.info("Creating hypothesis embedding based on breakdown...")
        await self.llm.get_hypothesis_embedding()

    async def get_stock_hypothesis_news_topics(
        self, topic_ids: List[str]
    ) -> Tuple[List[HypothesisNewsTopicInfo], List[CompanyNewsTopicInfo]]:
        # TODO: either `news_development_list`, or the output of `_get_strs_lookup` is fine
        news_topics = get_news_topics(topic_ids)
        topic_id_to_topic = {topic.topic_id: topic for topic in news_topics}

        sorted_topic_ids = self.ch.sort_news_topics_via_embeddings(
            news_topic_ids=list(topic_id_to_topic.keys()),
            embedding_vector=self.hypothesis.embedding,  # type: ignore
            embedding_model_id=TEXT_3_LARGE,
        )
        sorted_news_topics = [topic_id_to_topic[topic_id] for topic_id in sorted_topic_ids]

        relevant_topics: List[CompanyNewsTopicInfo] = []
        hypothesis_topics: List[HypothesisNewsTopicInfo] = []

        topic_worker_queue: asyncio.Queue = asyncio.Queue()
        worker_tasks: List[asyncio.Task] = []

        for _ in range(NUM_TOPIC_WORKERS):
            worker_tasks.append(
                asyncio.create_task(
                    self._hypothesis_topic_worker(topic_worker_queue, hypothesis_topics)
                )
            )

        logger.info("Searching hypothesis relevant topics...")
        for idx in range(0, len(sorted_news_topics), NEWS_TOPICS_BATCH_SIZE):
            batch_idx = idx // NEWS_TOPICS_BATCH_SIZE + 1
            logger.info(f"Processing batch {batch_idx} of news topics...")

            topics_batch = sorted_news_topics[idx : idx + NEWS_TOPICS_BATCH_SIZE]
            batch_size = len(topics_batch)
            relevant_topics_mask = await self.llm.check_hypothesis_relevant_topics(topics_batch)

            # add all relevant topics to the queue
            batch_relevant_topics = [
                topic for topic, is_related in zip(topics_batch, relevant_topics_mask) if is_related
            ]
            for topic in batch_relevant_topics:
                topic_worker_queue.put_nowait(topic)
            relevant_topics.extend(batch_relevant_topics)

            # now determine if we should stop processing
            # 0. If total relevant topics greater than threshold, we stop for now
            # 1. if we have processed `MAX_BATCHES` batches
            # 2. if the bottom `IRRELEVANT_TOPICS_THRESHOLD` topics are all irrelevant
            # 3. if there are less than `TOTAL_RELEVANT_TOPICS_THRESHOLD` relevant topics in the batch
            if len(relevant_topics) >= NUM_TOPICS_UB:
                logger.info(f"Reached {NUM_TOPICS_UB=}. Stopping process...")
                break
            elif batch_idx >= MAX_BATCHES:
                logger.info(f"Reached {MAX_BATCHES=}. Stopping process...")
                break
            elif len(batch_relevant_topics) < batch_size * TOTAL_RELEVANT_TOPICS_THRESHOLD:
                logger.info(
                    f"Batch {batch_idx} has only {len(relevant_topics)} relevant topics, less than required threshold. "
                    "Stopping process..."
                )
                break

            cutoff_pos = batch_size - int(batch_size * IRRELEVANT_TOPICS_THRESHOLD)
            if all(not is_relevant for is_relevant in relevant_topics_mask[cutoff_pos:]):
                logger.info(
                    f"Bottom {int(IRRELEVANT_TOPICS_THRESHOLD * 100)}% topics are irrelevant in batch {batch_idx}. "
                    "Stopping process..."
                )
                break

        # need join to make sure we are waiting until all the contents of the queue have been processed
        await topic_worker_queue.join()

        # need cancel because our workers run forever and don't have any other way to break out of their tasks
        for task in worker_tasks:
            task.cancel()

        # need gather to be sure all worker tasks are properly cancelled
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        logger.info(f"Found {len(hypothesis_topics)} relevant hypothesis news topics")

        # TODO: for now, we can assume that all the relevant topics are here, but later during updates
        # we may need to get them
        corresponding_topics = [topic_id_to_topic[topic.topic_id] for topic in relevant_topics]
        return hypothesis_topics, corresponding_topics

    async def _hypothesis_topic_worker(
        self,
        input_queue: asyncio.Queue,
        output_list: Union[List[HypothesisNewsTopicInfo], List[HypothesisEarningsTopicInfo]],
    ) -> None:
        while True:
            topic: CompanyNewsTopicInfo = await input_queue.get()
            result = await self.llm.create_hypothesis_topic(self.hypothesis, topic)
            if result:
                output_list.append(result)  # type: ignore
            input_queue.task_done()
