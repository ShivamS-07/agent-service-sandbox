import asyncio
import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union

from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.GPT.constants import TEXT_3_LARGE
from agent_service.io_types.text import (
    StockHypothesisCustomDocumentText,
    StockHypothesisEarningsSummaryPointText,
    StockHypothesisNewsDevelopmentText,
)
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.hypothesis.constants import (
    HORIZON_DELTA_LOOKUP,
    IRRELEVANT_TOPICS_THRESHOLD,
    MAX_BATCHES,
    NEWS_TOPICS_BATCH_SIZE,
    NUM_NEWS_TOPICS_FOR_SUMMARY,
    NUM_TOPIC_WORKERS,
    NUM_TOPICS_UB,
    PROPERTY,
    TOTAL_RELEVANT_TOPICS_THRESHOLD,
)
from agent_service.utils.hypothesis.hypothesis_ai import HypothesisAI
from agent_service.utils.hypothesis.types import (
    CompanyEarningsTopicInfo,
    CompanyNewsInfo,
    CompanyNewsTopicInfo,
    CustomDocTopicInfo,
    EarningsSummaryType,
    HypothesisEarningsTopicInfo,
    HypothesisInfo,
    HypothesisNewsTopicInfo,
)
from agent_service.utils.hypothesis.utils import (
    convert_to_news_groups,
    get_custom_document_news_from_documents,
    get_custom_document_news_topics,
    get_earnings_topics,
    get_hypothesis_match_chart,
    get_hypothesis_topic_weights,
    get_max_news_count_pair_across_stocks,
    get_news_from_topics,
    get_news_topics,
)
from agent_service.utils.postgres import get_psql

logger = logging.getLogger(__name__)


class HypothesisPipeline:
    def __init__(
        self,
        gbi_id: int,
        hypothesis_text: str,
        ref_time: Optional[datetime.datetime] = None,
        gpt_service_stub: Optional[GPTServiceStub] = None,
    ) -> None:
        self.ref_time = ref_time if ref_time else get_now_utc()

        self.pg = get_psql()
        self.ch = Clickhouse()

        self.hypothesis = self.create_hypothesis_info(gbi_id, hypothesis_text)

        self.llm = HypothesisAI(
            self.hypothesis, ref_time=self.ref_time, gpt_service_stub=gpt_service_stub
        )

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

        sorted_topic_ids = await self.ch.sort_news_topics_via_embeddings(
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
                    f"Batch {batch_idx} has only {len(batch_relevant_topics)} relevant topics,"
                    " less than required threshold. Stopping process..."
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
        corresponding_topics = [topic_id_to_topic[topic.topic_id] for topic in hypothesis_topics]
        return hypothesis_topics, corresponding_topics

    async def get_stock_hypothesis_earnings_topics(
        self, summary_ids: List[str]
    ) -> Tuple[List[HypothesisEarningsTopicInfo], List[CompanyEarningsTopicInfo]]:
        logger.info("Getting earnings topics from DB...")
        earnings_topics = get_earnings_topics(summary_ids)
        if len(earnings_topics) == 0:
            return [], []

        logger.info("Searching hypothesis relevant earnings topics...")
        relevant_topics_mask = await self.llm.check_hypothesis_relevant_topics(earnings_topics)

        logger.info("Creating hypothesis earnings topics...")
        tasks = [
            self.llm.create_hypothesis_topic(topic)
            for topic, is_related in zip(earnings_topics, relevant_topics_mask)
            if is_related
        ]
        hypothesis_topics: List[Optional[HypothesisEarningsTopicInfo]] = (
            await gather_with_concurrency(tasks, n=len(tasks))
        )
        filtered_hypothesis_topics = [topic for topic in hypothesis_topics if topic is not None]

        topic_id_to_topic = {
            (topic.topic_id, topic.summary_type, topic.summary_index): topic
            for topic in earnings_topics
        }
        corresponding_topics = [
            topic_id_to_topic[(topic.topic_id, topic.summary_type, topic.summary_index)]
            for topic in filtered_hypothesis_topics
        ]

        return filtered_hypothesis_topics, corresponding_topics

    async def get_stock_hypothesis_custom_document_topics(
        self, custom_document_news_ids: List[str]
    ) -> Tuple[List[HypothesisNewsTopicInfo], List[CustomDocTopicInfo], Dict[str, str]]:
        # TODO: either `news_development_list`, or the output of `_get_strs_lookup` is fine
        news_topics = get_custom_document_news_topics(custom_document_news_ids)
        topic_id_to_topic = {topic.topic_id: topic for topic in news_topics}
        topic_id_to_news_id = {topic.topic_id: topic.news_id for topic in news_topics}

        sorted_topic_ids = await self.ch.sort_news_topics_via_embeddings(
            news_topic_ids=list(topic_id_to_topic.keys()),
            embedding_vector=self.hypothesis.embedding,  # type: ignore
            embedding_model_id=TEXT_3_LARGE,
        )
        sorted_news_topics = [topic_id_to_topic[topic_id] for topic_id in sorted_topic_ids]

        relevant_topics: List[CustomDocTopicInfo] = []
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
                    f"Batch {batch_idx} has only {len(batch_relevant_topics)} relevant topics, "
                    "less than required threshold. Stopping process..."
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
        corresponding_topics = [topic_id_to_topic[topic.topic_id] for topic in hypothesis_topics]
        return hypothesis_topics, corresponding_topics, topic_id_to_news_id

    async def _hypothesis_topic_worker(
        self,
        input_queue: asyncio.Queue,
        output_list: Union[
            List[HypothesisNewsTopicInfo],
            List[HypothesisEarningsTopicInfo],
            List[CustomDocTopicInfo],
        ],
    ) -> None:
        while True:
            topic: CompanyNewsTopicInfo = await input_queue.get()
            result = await self.llm.create_hypothesis_topic(topic)
            if result:
                output_list.append(result)  # type: ignore
            input_queue.task_done()

    ################################################################################################
    # Match Score and Summary
    ################################################################################################
    async def calculate_match_score_and_generate_summary(
        self,
        news_developments: List[StockHypothesisNewsDevelopmentText],
        earnings_summary_points: List[StockHypothesisEarningsSummaryPointText],
        custom_documents: List[StockHypothesisCustomDocumentText],
    ) -> Tuple[
        float,
        str,
        List[StockHypothesisNewsDevelopmentText],
        List[StockHypothesisEarningsSummaryPointText],
        List[StockHypothesisCustomDocumentText],
    ]:
        """Calculate the match score and generate the hypothesis summary based on the news
        developments and earnings summary points.
        Note that the news developments and earnings summary points must have `history` field filled
        and the `explanation` and `score` will be used to create hypothesis topic objects
        """

        if not any((news_developments, earnings_summary_points, custom_documents)):
            return 0, "", [], [], []

        ref_time = get_now_utc()

        logger.info("Creating news and earnings topics from Text objects...")
        (
            news_topics,
            news_hypothesis_topics,
            news_groups,
        ) = self._convert_hypothesis_news_developments_to_topics(
            news_developments, ref_time=ref_time
        )
        (
            earnings_topics,
            earnings_hypothesis_topics,
        ) = await self._convert_hypothesis_earnings_summary_points_to_topics(
            earnings_summary_points, ref_time=ref_time
        )
        (
            custom_document_news_topics,
            custom_document_hypothesis_topics,
            custom_document_news_groups,
        ) = await self._convert_hypothesis_custom_documents_to_topics(
            custom_documents,
            ref_time=ref_time,
        )

        logger.info("Calculating match scores...")
        gbi_ids = list({t.gbi_id for t in news_topics} | {t.gbi_id for t in earnings_topics})
        max_count_pair = get_max_news_count_pair_across_stocks(gbi_ids)
        match_score = self.calculate_match_score(
            news_hypothesis_topics=news_hypothesis_topics,
            news_groups=news_groups,
            earnings_hypothesis_topics=earnings_hypothesis_topics,
            max_count_pair=max_count_pair,
        )

        logger.info("Generating hypothesis summary...")
        (
            summary,
            news_ref_hypo_topics,
            earnings_ref_hypo_topics,
            custom_document_ref_hypo_topics,
        ) = await self.generate_hypothesis_summary(
            self.hypothesis.hypothesis_breakdown[PROPERTY],  # type: ignore
            news_topics,
            news_hypothesis_topics,
            news_groups,
            earnings_topics,
            earnings_hypothesis_topics,
            custom_document_news_topics,
            custom_document_hypothesis_topics,
            custom_document_news_groups,
            max_count_pair,
            match_score,
            ref_time,
        )

        id_to_development = {development.id: development for development in news_developments}
        tup_to_point = {
            (p.summary_id, p.summary_idx, p.summary_type): p for p in earnings_summary_points
        }
        id_to_custom_doc = {doc.topic_id: doc for doc in custom_documents}

        ref_news_developments = [id_to_development[t.topic_id] for t in news_ref_hypo_topics]
        ref_earnings_points = [
            tup_to_point[(t.topic_id, t.summary_index, t.summary_type)]
            for t in earnings_ref_hypo_topics
        ]
        ref_custom_document_points = [
            id_to_custom_doc[t.topic_id] for t in custom_document_ref_hypo_topics
        ]

        return (
            match_score,
            summary,
            ref_news_developments,
            ref_earnings_points,
            ref_custom_document_points,
        )

    def _convert_hypothesis_news_developments_to_topics(
        self,
        news_developments: List[StockHypothesisNewsDevelopmentText],
        ref_time: datetime.datetime,
    ) -> Tuple[
        List[CompanyNewsTopicInfo], List[HypothesisNewsTopicInfo], List[List[CompanyNewsInfo]]
    ]:
        if not news_developments:
            return [], [], []

        if ref_time is None:
            ref_time = get_now_utc()

        topic_ids = [development.id for development in news_developments]

        logger.info("Getting news topics from DB...")
        news_topics = get_news_topics(topic_ids)

        news_topic_id_to_development = {
            development.id: development for development in news_developments
        }
        hypothesis_topics = []
        for news_topic in news_topics:
            development = news_topic_id_to_development[news_topic.topic_id]
            support_score = development.support_score.rescale(lb=-1, ub=1)
            hypothesis_topics.append(
                HypothesisNewsTopicInfo(
                    gbi_id=news_topic.gbi_id,
                    topic_id=news_topic.topic_id,
                    hypothesis_topic_supports=[(support_score, ref_time)],
                    hypothesis_topic_impacts=[],
                    hypothesis_topic_polarities=[],
                    hypothesis_topic_reasons=[(development.reason, ref_time)],
                )
            )

        logger.info("Getting all related news articles and grouping...")
        topic_ids = [development.id for development in news_developments]
        min_time = ref_time - HORIZON_DELTA_LOOKUP["4M"]
        news_infos = get_news_from_topics(topic_ids, min_time, ref_time)
        news_groups = convert_to_news_groups(hypothesis_topics, news_infos)

        return news_topics, hypothesis_topics, news_groups

    async def _convert_hypothesis_earnings_summary_points_to_topics(
        self,
        earnings_summary_points: List[StockHypothesisEarningsSummaryPointText],
        ref_time: datetime.datetime,
    ) -> Tuple[List[CompanyEarningsTopicInfo], List[HypothesisEarningsTopicInfo]]:
        if not earnings_summary_points:
            return [], []

        logger.info("Getting earnings topics from DB...")
        id_to_description = await StockHypothesisEarningsSummaryPointText._get_strs_lookup(
            earnings_summary_points
        )
        earnings_topics = []
        hypothesis_topics = []
        for point in earnings_summary_points:
            earnings_topics.append(
                CompanyEarningsTopicInfo(
                    gbi_id=point.stock_id.gbi_id,  # type: ignore
                    topic_id=point.summary_id,
                    topic_label="",  # placeholder, not used
                    topic_descriptions=[(id_to_description[point.id], ref_time)],
                    topic_polarities=[],
                    summary_index=point.summary_idx,
                    summary_type=EarningsSummaryType(point.summary_type),
                    summary_date=ref_time,  # placeholder, not used
                    topic_impacts=[],
                )
            )
            hypothesis_topics.append(
                HypothesisEarningsTopicInfo(
                    gbi_id=-1,
                    topic_id=point.summary_id,
                    summary_index=point.summary_idx,
                    summary_type=EarningsSummaryType(point.summary_type),
                    summary_date=ref_time,
                    hypothesis_topic_supports=[
                        (point.support_score.rescale(lb=-1, ub=1), ref_time)
                    ],
                    hypothesis_topic_impacts=[],
                    hypothesis_topic_polarities=[],
                    hypothesis_topic_reasons=[(point.reason, ref_time)],
                )
            )

        return earnings_topics, hypothesis_topics

    async def _convert_hypothesis_custom_documents_to_topics(
        self,
        custom_documents: List[StockHypothesisCustomDocumentText],
        ref_time: datetime.datetime,
    ) -> Tuple[
        List[CustomDocTopicInfo], List[HypothesisNewsTopicInfo], List[List[CompanyNewsInfo]]
    ]:
        if not custom_documents:
            return [], [], []

        if ref_time is None:
            ref_time = get_now_utc()

        logger.info("Getting news topics from DB...")
        news_id_to_doc = {document.id: document for document in custom_documents}
        news_topics = get_custom_document_news_topics(list(news_id_to_doc.keys()))
        # the news ID is the custom doc ID
        news_topic_id_to_custom_doc = {
            news_topic.topic_id: news_id_to_doc[news_topic.news_id] for news_topic in news_topics
        }

        hypothesis_topics = []
        for news_topic in news_topics:
            custom_document = news_topic_id_to_custom_doc[news_topic.topic_id]
            support_score = custom_document.support_score.rescale(lb=-1, ub=1)
            hypothesis_topics.append(
                HypothesisNewsTopicInfo(
                    gbi_id=news_topic.gbi_id,
                    topic_id=news_topic.topic_id,
                    hypothesis_topic_supports=[(support_score, ref_time)],
                    hypothesis_topic_impacts=[],
                    hypothesis_topic_polarities=[],
                    hypothesis_topic_reasons=[(custom_document.reason, ref_time)],
                )
            )

        logger.info("Getting all related news articles and grouping...")
        topic_ids = list(news_topic_id_to_custom_doc.keys())
        min_time = ref_time - HORIZON_DELTA_LOOKUP["4M"]
        custom_document_news_infos = get_custom_document_news_from_documents(
            topic_ids, min_time, ref_time
        )
        custom_document_news_groups = convert_to_news_groups(
            hypothesis_topics, custom_document_news_infos
        )
        return news_topics, hypothesis_topics, custom_document_news_groups

    def calculate_match_score(
        self,
        news_hypothesis_topics: List[HypothesisNewsTopicInfo],
        news_groups: List[List[CompanyNewsInfo]],
        earnings_hypothesis_topics: List[HypothesisEarningsTopicInfo],
        max_count_pair: Tuple[int, int],
    ) -> float:
        match_scores = get_hypothesis_match_chart(
            news_hypothesis_topics=news_hypothesis_topics,
            earnings_hypothesis_topics=earnings_hypothesis_topics,
            news_groups=news_groups,
            max_count_pair=max_count_pair,
            chart_horizon="3M",
            window_size="1M",
        )
        return match_scores[-1][1]

    async def generate_hypothesis_summary(
        self,
        property: str,
        news_topics: List[CompanyNewsTopicInfo],
        news_hypothesis_topics: List[HypothesisNewsTopicInfo],
        news_groups: List[List[CompanyNewsInfo]],
        earnings_topics: List[CompanyEarningsTopicInfo],
        earnings_hypothesis_topics: List[HypothesisEarningsTopicInfo],
        custom_document_news_topics: List[CustomDocTopicInfo],
        custom_document_hypothesis_topics: List[HypothesisNewsTopicInfo],
        custom_document_news_groups: List[List[CompanyNewsInfo]],
        max_count_pair: Tuple[int, int],
        match_score: float,
        ref_time: datetime.datetime,
    ) -> Tuple[
        str,
        List[HypothesisNewsTopicInfo],
        List[HypothesisEarningsTopicInfo],
        List[HypothesisNewsTopicInfo],
    ]:
        if not any((news_topics, earnings_topics, custom_document_news_topics)):
            return "", [], [], []

        # Process hypothesis for news
        logger.info("Reordering news topics based on weights and joining as text...")
        topic_weights = get_hypothesis_topic_weights(
            news_hypothesis_topics, news_groups, max_count_pair, ref_time=ref_time
        )
        news_topic_pairs: List[Tuple[HypothesisNewsTopicInfo, CompanyNewsTopicInfo]] = [
            (hypo_topic, topic)
            for hypo_topic, topic, _ in sorted(
                zip(news_hypothesis_topics, news_topics, topic_weights),
                key=lambda x: abs(x[-1]),
                reverse=True,
            )
        ]
        news_topic_list: List[str] = []
        for i, (hypo_topic, topic) in enumerate(news_topic_pairs[:NUM_NEWS_TOPICS_FOR_SUMMARY]):
            news_topic_list.append(
                f"Topic {i + 1}\n"
                f"Topic Description: {topic.get_latest_topic_description()}\n"
                f"Topic Connection: {hypo_topic.get_latest_reason()}"
            )
        news_topics_str = "\n\n".join(news_topic_list)

        # Process hypothesis for earnings
        earnings_topic_list: List[str] = []
        for i, (topic, hypo_topic) in enumerate(zip(earnings_topics, earnings_hypothesis_topics)):
            earnings_topic_list.append(
                f"Topic {i + 1}\n"
                f"Topic Description: {topic.get_latest_topic_description()}\n"
                f"Topic Connection: {hypo_topic.get_latest_reason()}"
            )
        earnings_main_topics_str = "\n\n".join(earnings_topic_list)

        # Process hypothesis for custom documents
        custom_document_topic_weights = get_hypothesis_topic_weights(
            custom_document_hypothesis_topics,
            custom_document_news_groups,
            max_count_pair,
            ref_time=ref_time,
        )
        custom_document_news_topic_pairs: List[
            Tuple[HypothesisNewsTopicInfo, CustomDocTopicInfo]
        ] = [
            (hypo_topic, topic)
            for hypo_topic, topic, _ in sorted(
                zip(
                    custom_document_hypothesis_topics,
                    custom_document_news_topics,
                    custom_document_topic_weights,
                ),
                key=lambda x: abs(x[-1]),
                reverse=True,
            )
        ]
        custom_document_topic_list: List[str] = []
        for i, (hypo_topic, topic) in enumerate(
            custom_document_news_topic_pairs[:NUM_NEWS_TOPICS_FOR_SUMMARY]
        ):
            custom_document_topic_list.append(
                f"Topic {i + 1}\n"
                f"Topic Description: {topic.get_latest_topic_description()}\n"
                f"Topic Connection: {hypo_topic.get_latest_reason()}"
            )
        custom_docs_news_topics_str = "\n\n".join(custom_document_topic_list)

        # Join everything together
        summary_dict = await self.llm.write_hypothesis_summary(
            property,
            match_score,
            news_topics_str,
            earnings_main_topics_str,
            custom_docs_news_topics_str,
        )
        summary: str = summary_dict["summary"]  # type: ignore

        news_ref_idxs: List[int] = summary_dict.get("news_references", [])  # type: ignore
        news_ref_hypo_topics = [news_topic_pairs[idx - 1][0] for idx in news_ref_idxs]

        earnings_ref_idxs: List[int] = summary_dict.get("earnings_references", [])  # type: ignore
        earnings_ref_hypo_topics = [
            earnings_hypothesis_topics[idx - 1] for idx in earnings_ref_idxs
        ]

        custom_document_news_ref_idxs: List[int] = summary_dict.get(  # type: ignore
            "custom_doc_news_references", []
        )
        custom_document_news_ref_hypo_topics = [
            custom_document_news_topic_pairs[idx - 1][0] for idx in custom_document_news_ref_idxs
        ]

        return (
            summary,
            news_ref_hypo_topics,
            earnings_ref_hypo_topics,
            custom_document_news_ref_hypo_topics,
        )
