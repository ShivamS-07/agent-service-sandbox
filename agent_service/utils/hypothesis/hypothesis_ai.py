import datetime
import json
import logging
from typing import Dict, List, Optional, Union

from gpt_service_proto_v1.service_grpc import GPTServiceStub

from agent_service.GPT.constants import DEFAULT_CHEAP_MODEL, SONNET, TEXT_3_LARGE
from agent_service.GPT.requests import GPT
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.date_utils import get_now_utc
from agent_service.utils.hypothesis.constants import (
    CONTRADICTS,
    EXPLANATION,
    IMPACT,
    MIXED_STR,
    NEUTRAL_CUTOFF,
    OPPOSE_STR,
    POLARITY,
    PROPERTY,
    RATIONALE,
    RELATION,
    STRENGTH,
    SUPPORT_DEGREE_LOOKUP,
    SUPPORT_LOOKUP,
    SUPPORT_STR,
    SUPPORTS,
)
from agent_service.utils.hypothesis.prompts import (
    HYPOTHESIS_EXPLANATION_PROMPT,
    HYPOTHESIS_RELEVANT_PROMPT,
    HYPOTHESIS_SUMMARY_CUSTOM_DOCS_NEWS_REFERENCE_TEMPLATE,
    HYPOTHESIS_SUMMARY_CUSTOM_DOCS_NEWS_TEMPLATE,
    HYPOTHESIS_SUMMARY_EARNINGS_REFERENCE_TEMPLATE,
    HYPOTHESIS_SUMMARY_EARNINGS_TEMPLATE,
    HYPOTHESIS_SUMMARY_MAIN_PROMPT,
    HYPOTHESIS_SUMMARY_NEWS_REFERENCE_TEMPLATE,
    HYPOTHESIS_SUMMARY_NEWS_TEMPLATE,
    HYPOTHESIS_SUMMARY_SYS_PROMPT,
    HYPOTHESIS_TOPIC_ANALYSIS_MAIN_PROMPT,
    HYPOTHESIS_TOPIC_ANALYSIS_SYS_PROMPT,
)
from agent_service.utils.hypothesis.types import (
    CompanyEarningsTopicInfo,
    CompanyNewsTopicInfo,
    CustomDocTopicInfo,
    EarningsSummaryType,
    HypothesisEarningsTopicInfo,
    HypothesisInfo,
    HypothesisNewsTopicInfo,
    NewsImpact,
    Polarity,
)
from agent_service.utils.prompt_utils import FilledPrompt
from agent_service.utils.string_utils import clean_to_json_if_needed

logger = logging.getLogger(__name__)


class HypothesisAI:
    def __init__(
        self,
        hypothesis_info: HypothesisInfo,
        context: Optional[Dict[str, str]] = None,
        ref_time: Optional[datetime.datetime] = None,
        gpt_service_stub: Optional[GPTServiceStub] = None,
    ) -> None:
        self.gpt_smart = GPT(model=SONNET, context=context, gpt_service_stub=gpt_service_stub)
        self.gpt_cheap = GPT(
            model=DEFAULT_CHEAP_MODEL, context=context, gpt_service_stub=gpt_service_stub
        )

        self.hypothesis = hypothesis_info
        self.ref_time = ref_time if ref_time else get_now_utc()

    async def get_hypothesis_breakdown(self) -> None:
        # TODO: In the warren's context it's different from the original NLP logic in the aspects:
        # 1. `hypothesis_text` becomes the `property`
        # 2. `company_name` is not used in the prompt
        # 3. We currently don't need `polarity` in the breakdown

        # main_prompt = HYPOTHESIS_PROPERTY_PROMPT.format(
        #     company_name=self.hypothesis.company_name,
        #     hypothesis=self.hypothesis.hypothesis_text,
        # )
        # property, polarity = (
        #     (
        #         await self.gpt_smart.do_chat_w_sys_prompt(
        #             main_prompt, sys_prompt=FilledPrompt(filled_prompt=""), max_tokens=15
        #         )
        #     )
        #     .strip()
        #     .split("\n")
        # )

        main_prompt = HYPOTHESIS_EXPLANATION_PROMPT.format(property=self.hypothesis.hypothesis_text)
        explanation = await self.gpt_cheap.do_chat_w_sys_prompt(
            main_prompt, sys_prompt=FilledPrompt(filled_prompt=""), max_tokens=200
        )

        self.hypothesis.hypothesis_breakdown = {
            PROPERTY: self.hypothesis.hypothesis_text,
            EXPLANATION: explanation,
            POLARITY: "Neutral",
        }

    async def get_hypothesis_embedding(self) -> None:
        if self.hypothesis.hypothesis_breakdown is None:
            return

        embedding = await self.gpt_smart.embed_text(
            self.hypothesis.hypothesis_breakdown[PROPERTY]
            + "\n"
            + self.hypothesis.hypothesis_breakdown[EXPLANATION],
            embedding_model=TEXT_3_LARGE,
        )
        self.hypothesis.embedding = embedding

    async def check_hypothesis_relevant_topics(
        self,
        topics: Union[
            List[CompanyNewsTopicInfo], List[CompanyEarningsTopicInfo], List[CustomDocTopicInfo]
        ],
    ) -> List[bool]:
        tasks = []
        if self.hypothesis.hypothesis_breakdown is None:
            return [False] * len(topics)

        for topic in topics:
            # For now we just check the most recent description
            tasks.append(
                self._check_relevance_for_topic(
                    self.hypothesis.hypothesis_breakdown[PROPERTY],
                    self.hypothesis.hypothesis_breakdown[EXPLANATION],
                    topic.topic_label,
                    topic.get_latest_topic_description(),  # type: ignore
                )
            )
        return await gather_with_concurrency(tasks, n=len(tasks))

    async def _check_relevance_for_topic(
        self,
        hypothesis_property: str,
        property_explanation: str,
        topic_title: str,
        topic_description: str,
    ) -> bool:
        topic_rep = topic_title + "\n" + topic_description
        main_prompt = HYPOTHESIS_RELEVANT_PROMPT.format(
            property=hypothesis_property,
            explanation=property_explanation,
            topic=topic_rep,
            company_name=self.hypothesis.company_name,
        )
        result = await self.gpt_cheap.do_chat_w_sys_prompt(
            main_prompt, sys_prompt=FilledPrompt(filled_prompt=""), max_tokens=2
        )
        return result.lower().startswith("yes")

    async def create_hypothesis_topic(
        self, topic: Union[CompanyNewsTopicInfo, CompanyEarningsTopicInfo]
    ) -> Optional[Union[HypothesisNewsTopicInfo, HypothesisEarningsTopicInfo]]:
        hypothesis_breakdown = self.hypothesis.hypothesis_breakdown or {}

        main_prompt = HYPOTHESIS_TOPIC_ANALYSIS_MAIN_PROMPT.format(
            company_name=self.hypothesis.company_name,
            company_description=self.hypothesis.company_description,
            topic_label=topic.topic_label,
            topic_description=topic.get_latest_topic_description(),  # type: ignore
            topic_impact=topic.get_latest_topic_impact(NewsImpact.low),  # type: ignore
            topic_polarity=topic.get_latest_topic_polarity(),  # type: ignore
            hypothesis=self.hypothesis.hypothesis_text,
            property=hypothesis_breakdown.get(PROPERTY, ""),
            explanation=hypothesis_breakdown.get(EXPLANATION, ""),
            hypothesis_polarity=hypothesis_breakdown.get(POLARITY, "Neutral"),
        )
        result = await self.gpt_smart.do_chat_w_sys_prompt(
            main_prompt, HYPOTHESIS_TOPIC_ANALYSIS_SYS_PROMPT.format()
        )

        try:
            result_json = json.loads(clean_to_json_if_needed(result))
        except json.JSONDecodeError as e:
            logger.error(
                f"Could not decode json for `{topic.topic_label}`, Error: {e}, GPT ouput: {result}"
            )
            return None

        if not result_json:
            return None

        try:
            if SUPPORTS in result_json[RELATION]:
                support = SUPPORT_LOOKUP[SUPPORTS]
            elif CONTRADICTS in result_json[RELATION]:
                support = SUPPORT_LOOKUP[CONTRADICTS]
            else:
                # Sometimes the LLM only realizes it's not relevant after it writes the rationale
                # This gives us another chance to throw out irrelevant articles
                return None
            degree = SUPPORT_DEGREE_LOOKUP[result_json[STRENGTH].lower()]
            support_num = support * degree  # [-1, -0.5, -0.1, 0.1, 0.5, 1]
            description_time = topic.topic_descriptions[-1][1]

            topic_dict = {
                "gbi_id": self.hypothesis.gbi_id,
                "topic_id": topic.topic_id,
                "hypothesis_topic_supports": [(support_num, description_time)],
                "hypothesis_topic_reasons": [(result_json[RATIONALE], description_time)],
                "hypothesis_topic_polarities": [
                    (Polarity[result_json[POLARITY].lower()], description_time)
                ],
                "hypothesis_topic_impacts": [
                    (NewsImpact[result_json[IMPACT].lower()], description_time)
                ],
            }

            if isinstance(topic, CompanyNewsTopicInfo):
                return HypothesisNewsTopicInfo(**topic_dict)  # type: ignore
            elif isinstance(topic, CustomDocTopicInfo):
                return HypothesisNewsTopicInfo(**topic_dict)  # type: ignore
            elif isinstance(topic, CompanyEarningsTopicInfo):
                # Currently not involving the questions portion whatsoever so we can add this
                # to all of our generated hypothesis_topics, will need to modify this if we scope
                # in Question topics
                topic_dict["summary_index"] = topic.summary_index
                topic_dict["summary_type"] = topic.summary_type
                topic_dict["summary_date"] = topic.summary_date
                if topic.summary_type == EarningsSummaryType.PEERS:
                    topic_dict["peer_gbi_id"] = topic.peer_company_gbi_id
                return HypothesisEarningsTopicInfo(**topic_dict)  # type: ignore

        except KeyError as e:
            logger.error(
                f"Missing expected key when creating hypothesis topic `{topic.topic_label}`,"
                f"Error: {e}, GPT ouput: {result}"
            )
        return None

    async def write_hypothesis_summary(
        self,
        property: str,
        match_score: float,
        news_topics_str: str,
        earnings_main_topics_str: str,
        custom_docs_news_topics_str: str,
    ) -> Dict[str, Union[str, List[int]]]:
        """Generate a summary of the hypothesis analysis.

        Args:
            property (str): A sentence describing the hypothesis statement, right now it's the same
                as the hypothesis text.
            match_score (float): how much the hypothesis matches news and earnings topics, -1 to 1
            news_topics_str (str): A joined string of news topics that are relevant to the
                hypothesis. If not news topics, it should be an empty string.
            earnings_main_topics_str (str): A joined string of earnings topics that are relevant to
                the hypothesis. If not earnings topics, it should be an empty string.

        Returns:
            Dict[str, Union[str, List[int]]]: {
                "summary": str,
                "news_topics_references": List[int],  # key is optional
                "earnings_topics_references": List[int]  # key is optional
            }
        """
        if match_score > NEUTRAL_CUTOFF:
            conclusion = SUPPORT_STR
        elif match_score < -NEUTRAL_CUTOFF:
            conclusion = OPPOSE_STR
        else:
            conclusion = MIXED_STR

        if news_topics_str:
            news_str = HYPOTHESIS_SUMMARY_NEWS_TEMPLATE.format(news_topics=news_topics_str)
            news_ref = HYPOTHESIS_SUMMARY_NEWS_REFERENCE_TEMPLATE
        else:
            news_str = ""
            news_ref = ""

        if earnings_main_topics_str:
            earnings_str = HYPOTHESIS_SUMMARY_EARNINGS_TEMPLATE.format(
                earnings_main_topics=earnings_main_topics_str
            )
            earnings_ref = HYPOTHESIS_SUMMARY_EARNINGS_REFERENCE_TEMPLATE
        else:
            earnings_str = ""
            earnings_ref = ""

        if custom_docs_news_topics_str:
            custom_doc_news_str = HYPOTHESIS_SUMMARY_CUSTOM_DOCS_NEWS_TEMPLATE.format(
                custom_docs_news_topics=custom_docs_news_topics_str,
            )
            custom_doc_news_ref = HYPOTHESIS_SUMMARY_CUSTOM_DOCS_NEWS_REFERENCE_TEMPLATE
        else:
            custom_doc_news_str = ""
            custom_doc_news_ref = ""

        main_prompt = HYPOTHESIS_SUMMARY_MAIN_PROMPT.format(
            property=property,
            conclusion=conclusion,
            news_str=news_str,
            earnings_str=earnings_str,
            custom_doc_news_str=custom_doc_news_str,
            news_ref=news_ref,
            earnings_ref=earnings_ref,
            custom_doc_news_ref=custom_doc_news_ref,
        )

        json_str = await self.gpt_smart.do_chat_w_sys_prompt(
            main_prompt, HYPOTHESIS_SUMMARY_SYS_PROMPT.format(), max_tokens=300
        )
        return json.loads(clean_to_json_if_needed(json_str))
