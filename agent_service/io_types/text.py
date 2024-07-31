from __future__ import annotations

import datetime
import json
from collections import defaultdict
from itertools import chain
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)
from uuid import uuid4

import mdutils
from pydantic import Field
from typing_extensions import Self

from agent_service.external.custom_data_svc_client import get_custom_doc_articles_info
from agent_service.io_type_utils import (
    Citation,
    ComplexIOBase,
    IOType,
    Score,
    ScoreOutput,
    io_type,
)
from agent_service.io_types.citations import (
    CitationOutput,
    CompanyFilingCitationOutput,
    CustomDocumentCitationOutput,
    NewsArticleCitationOutput,
    NewsDevelopmentCitationOutput,
    TextCitationOutput,
    ThemeCitationOutput,
)
from agent_service.io_types.output import Output, OutputType
from agent_service.io_types.stock import StockID
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.sec.constants import LINK_TO_FILING_DETAILS
from agent_service.utils.sec.sec_api import SecFiling

TextIDType = Union[str, int]


@io_type
class Text(ComplexIOBase):
    id: TextIDType = Field(default_factory=lambda: str(uuid4()))
    val: str = ""
    text_type: ClassVar[str] = "Misc"
    stock_id: Optional[StockID] = None
    timestamp: Optional[datetime.datetime] = None

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Text):
            return self.id == other.id
        return False

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        # Get the citations for the current Text object, as well as any
        # citations from "child" (input) texts.
        tasks = [
            Citation.resolve_all_citations(citations=self.get_all_citations(), db=pg),
            self.get_citations_for_output(texts=[TextCitation(source_text=self)], db=pg),
        ]
        outputs = await gather_with_concurrency(tasks)
        citations = outputs[0] + outputs[1]
        text = await self.get()
        return TextOutput(
            val=text.val,
            title=title,
            citations=citations,
            score=ScoreOutput.from_entry_list(self.history),
        )

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        score = ScoreOutput.from_entry_list(self.history)
        text = await self.get()
        if not score:
            return f"<Text: {text.val}>"
        else:
            return f"<Text Score(s) (0 - 1 scale): {score.to_gpt_input()}, Text: {text.val}>"

    @staticmethod
    def _to_string_recursive(val: IOType) -> IOType:
        if isinstance(val, list):
            return [Text._to_string_recursive(v) for v in val]
        elif isinstance(val, dict):
            return {str(k): Text._to_string_recursive(v) for k, v in val.items()}
        elif isinstance(val, Text):
            return val.val
        else:
            return str(val)

    @staticmethod
    def from_io_type(val: IOType) -> Text:
        """
        Given an arbitrary IOType value, return it as a rich Text object
        """
        try:
            # Try to convert data to markdown, otherwise just return as a raw string
            if isinstance(val, list):
                md = mdutils.MdUtils(file_name="", title="")
                md.new_list(items=Text._to_string_recursive(val))
                text = md.get_md_text().strip()
            elif isinstance(val, dict):
                md = mdutils.MdUtils(file_name="", title="")
                table_cols = 2
                table_rows = len(val)
                table_values = [str(item) for tup in val.items() for item in tup]
                md.new_table(columns=table_cols, rows=table_rows, text=table_values)
                text = md.get_md_text().strip()
            else:
                text = str(val)
        except Exception:
            text = str(val)

        return Text(val=text)

    @classmethod
    async def get_all_strs(
        cls,
        text: Union[Text, TextGroup, List[Any], Dict[Any, Any]],
        include_header: bool = False,
        text_group_numbering: bool = False,
    ) -> Union[str, List[Any], Dict[Any, Any]]:
        # For any possible configuration of Texts or TextGroups in Lists, this converts that
        # configuration to the corresponding strings in list, in class specific batches
        # TextGroups become a single string here

        def convert_to_ids_and_categorize(
            text: Union[Text, TextGroup, List[Any], Dict[Any, Any]],
            categories: Dict[Type[Text], List[Text]],
        ) -> Union[TextIDType, TextGroup, List[Any], Dict[Any, Any]]:
            """
            Convert a structure of texts (e.g. nested lists, dicts, etc.) into an
            identical structure of text ID's, while also keeping track of all
            texts of each type.
            """
            if isinstance(text, list):
                return [convert_to_ids_and_categorize(sub_text, categories) for sub_text in text]
            elif isinstance(text, dict):
                # we assume texts are only values, not keys
                return {
                    key: convert_to_ids_and_categorize(value, categories)
                    for key, value in text.items()
                }
            else:
                if isinstance(text, TextGroup):
                    for subtext in text.val:
                        categories[type(subtext)].append(subtext)
                    return text
                else:
                    categories[type(text)].append(text)
                    return text.id

        categories: Dict[Type[Text], List[Text]] = defaultdict(list)
        # identical structure to input texts, but as IDs
        texts_as_ids = convert_to_ids_and_categorize(text, categories)
        if include_header:
            timestamp_lookup = {}
            for cat_texts in categories.values():
                for text in cat_texts:
                    timestamp_lookup[text.id] = text.timestamp
        else:
            timestamp_lookup = None
        strs_lookup: Dict[TextIDType, str] = {}
        # For every subclass of Text, fetch data
        lookups: List[Dict[TextIDType, str]] = await gather_with_concurrency(
            [
                textclass.get_strs_lookup(texts, timestamp_lookup=timestamp_lookup)
                for textclass, texts in categories.items()
            ]
        )
        for lookup in lookups:
            strs_lookup.update(lookup)

        def convert_ids_to_strs(
            strs_lookup: Dict[TextIDType, str],
            id_rep: Union[TextIDType, TextGroup, List[Any], Dict[Any, Any]],
        ) -> Union[str, List[Any], Dict[Any, Any]]:
            """
            Take the structure of ID lists, and map back into actual strings.
            """
            if isinstance(id_rep, dict):
                return {
                    key: convert_ids_to_strs(strs_lookup, value) for key, value in id_rep.items()
                }
            if isinstance(id_rep, list):
                return [convert_ids_to_strs(strs_lookup, sub_id_rep) for sub_id_rep in id_rep]
            if isinstance(id_rep, TextGroup):
                return id_rep.convert_to_str(strs_lookup, text_group_numbering)
            else:
                return strs_lookup[id_rep]

        return convert_ids_to_strs(strs_lookup, texts_as_ids)

    @classmethod
    async def get_strs_lookup(
        cls,
        texts: List[Self],
        timestamp_lookup: Optional[Dict[TextIDType, Optional[datetime.datetime]]] = None,
    ) -> Dict[TextIDType, str]:
        strs_lookup = await cls._get_strs_lookup(texts)
        if timestamp_lookup is not None:
            for id, val in strs_lookup.items():
                if id in timestamp_lookup and timestamp_lookup[id] is not None:
                    date_str = f"Date: {timestamp_lookup[id].date()}\n"  # type: ignore
                else:
                    date_str = ""
                strs_lookup[id] = f"Text type: {cls.text_type}\n{date_str}Text:\n{val}"
        return strs_lookup

    @classmethod
    async def _get_strs_lookup(cls, texts: List[Self]) -> Dict[TextIDType, str]:
        return {text.id: text.val for text in texts}

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        """
        Given a list of texts of this type, return a list of citations related
        the the input texts.
        Each input text citation may map to one or more output citations, so we
        return a list of output citations.
        """
        return []

    async def get(self) -> Text:
        """
        For an instance of a 'Text' subclass, resolve and return it as a standard Text object.
        """
        if not self.val:
            # resolve the text if necessary
            lookup = await self.get_strs_lookup([self])
            return Text(val=lookup[self.id])
        else:
            return Text(val=self.val)

    def reset_value(self) -> None:
        self.val = ""  # reset the value to avoid spamming DB


@io_type
class ProfileText(Text):
    # val will store the profile string
    importance_score: float


@io_type
class NewsText(Text):
    pass


@io_type
class StockText(Text):
    # stock_id is mandatory, or else no output will be yielded.
    stock_id: Optional[StockID] = None


@io_type
class EarningsPeersText(StockText):
    affecting_stock_id: Optional[StockID] = None
    year: int
    quarter: int


@io_type
class StockNewsDevelopmentText(NewsText, StockText):
    id: str
    text_type: ClassVar[str] = "News Development Summary"

    @classmethod
    async def _get_strs_lookup(
        cls,
        news_topics: List[StockNewsDevelopmentText],
    ) -> Dict[TextIDType, str]:  # type: ignore
        sql = """
        SELECT topic_id::TEXT, topic_label, (topic_descriptions->-1->>0)::TEXT AS description
        FROM nlp_service.stock_news_topics
        WHERE topic_id = ANY(%(topic_ids)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"topic_ids": [topic.id for topic in news_topics]})
        return {row["topic_id"]: f"{row['topic_label']}: {row['description']}" for row in rows}

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        sql = """
        SELECT snt.topic_id::TEXT, topic_label, updated_at, (topic_descriptions->-1->>0)::TEXT AS summary,
               COUNT(sn.news_id) AS num_articles
        FROM nlp_service.stock_news_topics snt
        JOIN nlp_service.stock_news sn ON sn.topic_id = snt.topic_id
        WHERE sn.topic_id = ANY(%(topic_ids)s)
        GROUP BY snt.topic_id
        """
        params = {"topic_ids": [text.source_text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return [
            NewsDevelopmentCitationOutput(
                id=row["topic_id"],
                name=row["topic_label"],
                summary=row["summary"],
                last_updated_at=row["updated_at"],
                num_articles=row["num_articles"],
            )
            for row in rows
        ]


@io_type
class StockHypothesisNewsDevelopmentText(StockNewsDevelopmentText):
    """
    Subclass from `StockNewsDevelopmentText`, stores the explanation and score for a hypothesis
    in `history` field
    """

    text_type: ClassVar[str] = "Hypothesis News Development Summary"

    support_score: Score
    reason: str


@io_type
class StockNewsDevelopmentArticlesText(NewsText, StockText):
    id: str
    text_type: ClassVar[str] = "News Article Summary"

    @classmethod
    async def _get_strs_lookup(
        cls, news_topics: List[StockNewsDevelopmentArticlesText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        from agent_service.utils.postgres import get_psql

        sql = """
            SELECT news_id::TEXT, headline, summary
            FROM nlp_service.stock_news
            WHERE news_id = ANY(%(news_ids)s)
        """
        rows = get_psql().generic_read(sql, {"news_ids": [topic.id for topic in news_topics]})
        return {row["news_id"]: f"{row['headline']}:\n{row['summary']}" for row in rows}

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        sql = """
        SELECT news_id::TEXT, url, domain_url, headline, published_at
        FROM nlp_service.stock_news sn
        JOIN nlp_service.news_sources ns ON ns.source_id = sn.source_id
        WHERE sn.news_id = ANY(%(news_ids)s)
        """
        params = {"news_ids": [text.source_text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return [
            NewsArticleCitationOutput(
                id=row["news_id"],
                name=row["domain_url"],
                link=row["url"],
                summary=row["headline"],
                last_updated_at=row["published_at"],
            )
            for row in rows
        ]


@io_type
class NewsPoolArticleText(NewsText):
    id: str
    text_type: ClassVar[str] = "News Article Summary"

    @classmethod
    async def _get_strs_lookup(cls, news_pool: List[NewsPoolArticleText]) -> Dict[str, str]:  # type: ignore
        sql = """
        SELECT news_id::TEXT, headline::TEXT, summary::TEXT
        FROM nlp_service.news_pool
        WHERE news_id = ANY(%(news_ids)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"news_ids": [topic.id for topic in news_pool]})
        return {row["news_id"]: f"{row['headline']}:\n{row['summary']}" for row in rows}

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        sql = """
        SELECT news_id::TEXT, url, domain_url, headline, published_at
        FROM nlp_service.news_pool np
        JOIN nlp_service.news_sources ns ON ns.source_id = np.source_id
        WHERE np.news_id = ANY(%(news_ids)s)
        """
        params = {"news_ids": [text.source_text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return [
            NewsArticleCitationOutput(
                id=row["news_id"],
                name=row["domain_url"],
                link=row["url"],
                summary=row["headline"],
                last_updated_at=row["published_at"],
            )
            for row in rows
        ]


@io_type
class CustomDocumentSummaryText(StockText):
    id: str
    requesting_user: str
    text_type: ClassVar[str] = "User Document Summary"

    @classmethod
    async def _get_strs_lookup(cls, articles_text: List[CustomDocumentSummaryText]) -> Dict[str, str]:  # type: ignore
        # group by user id - we are assuming that anyone who is authed for the agent
        # has priv to see any documents utilized by the agent.
        articles_text_by_user: Dict[str, List[CustomDocumentSummaryText]] = defaultdict(list)
        for doc in articles_text:
            articles_text_by_user[doc.requesting_user].append(doc)

        texts: Dict[str, str] = {}
        for user, articles in articles_text_by_user.items():
            article_info = await get_custom_doc_articles_info(
                user, [article.id for article in articles]
            )
            for id, chunk_info in dict(article_info.file_chunk_info).items():
                texts[id] = f"{chunk_info.headline}:\n{chunk_info.summary}"
        return texts

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        # group by user id - we are assuming that anyone who is authed for the agent
        # has priv to see any documents utilized by the agent.
        text_citation_by_user: Dict[str, List[TextCitation]] = defaultdict(list)
        text_id_map = {text.source_text.id: text for text in texts}
        for cit in texts:
            text_citation_by_user[cit.source_text.requesting_user].append(cit)  # type: ignore

        output_citations = []
        # TODO this will need to change once custom docs have better citation
        # support. Currently we just cite ALL chunks in the file.
        for user, citations in text_citation_by_user.items():
            article_info = await get_custom_doc_articles_info(
                user, [cit.source_text.id for cit in citations]
            )
            for chunk_id, chunk_info in dict(article_info.file_chunk_info).items():
                file_paths = list(chunk_info.file_paths)
                if len(file_paths) == 0:
                    citation_name = chunk_info.file_id
                else:
                    # pick an arbitrary custom file path
                    citation_name = file_paths[0]

                chunk_cit = text_id_map.get(chunk_info.chunk_id)
                output_citations.append(
                    CustomDocumentCitationOutput(
                        id=chunk_id,
                        name=f"User Document: {citation_name}",
                        last_updated_at=chunk_info.upload_time.ToDatetime(),
                        custom_doc_id=chunk_info.file_id,
                        inline_offset=chunk_cit.citation_text_offset if chunk_cit else None,
                    )
                )
        return output_citations


@io_type
class StockHypothesisCustomDocumentText(CustomDocumentSummaryText):
    """
    Subclass from `CustomDocumentSummaryText`, stores the explanation and score for a hypothesis
    in `history` field.

    Since these documents are external to NLP service, the topic ID is not the primary ID key
    but rather the news_id, therefore, we must augment this object to have an additional topic_id.
    """

    text_type: ClassVar[str] = "Hypothesis Custom Document Summary"

    topic_id: str
    support_score: Score
    reason: str


@io_type
class ThemeText(Text):
    id: str
    text_type: ClassVar[str] = "Theme Description"

    @classmethod
    async def _get_strs_lookup(cls, themes: List[ThemeText]) -> Dict[str, str]:  # type: ignore
        sql = """
        SELECT theme_id::TEXT, theme_description::TEXT AS description
        FROM nlp_service.themes
        WHERE theme_id = ANY(%(theme_id)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"theme_id": [topic.id for topic in themes]})
        return {row["theme_id"]: row["description"] for row in rows}

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        sql = """
        SELECT theme_id::TEXT, theme_name::TEXT AS name, theme_description, last_modified
        FROM nlp_service.themes
        WHERE theme_id = ANY(%(theme_id)s)
        """
        rows = await db.generic_read(sql, {"theme_id": [text.source_text.id for text in texts]})
        return [
            ThemeCitationOutput(
                id=row["theme_id"],
                name="Theme: " + row["name"],
                summary=row["theme_description"],
                last_updated_at=row["last_modified"],
            )
            for row in rows
        ]


@io_type
class ThemeNewsDevelopmentText(NewsText):
    id: str
    text_type: ClassVar[str] = "News Development Summary"

    @classmethod
    async def _get_strs_lookup(cls, themes: List[ThemeNewsDevelopmentText]) -> Dict[str, str]:  # type: ignore
        sql = """
        SELECT development_id::TEXT, label::TEXT, description::TEXT
        FROM nlp_service.theme_developments
        WHERE development_id = ANY(%(development_id)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"development_id": [topic.id for topic in themes]})
        return {row["development_id"]: f"{row['label']}:\n{row['description']}" for row in rows}

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        sql = """
        SELECT development_id::TEXT, label::TEXT, description, development_time
        FROM nlp_service.theme_developments
        WHERE development_id = ANY(%(development_id)s)
        """
        rows = await db.generic_read(
            sql, {"development_id": [text.source_text.id for text in texts]}
        )
        return [
            NewsDevelopmentCitationOutput(
                id=row["development_id"],
                name="News Development: " + row["label"],
                summary=row["description"],
                last_updated_at=row["development_time"],
            )
            for row in rows
        ]


@io_type
class ThemeNewsDevelopmentArticlesText(NewsText):
    id: str
    text_type: ClassVar[str] = "News Article Summary"

    @classmethod
    async def _get_strs_lookup(
        cls, developments: List[ThemeNewsDevelopmentArticlesText]
    ) -> Dict[Union[int, str], str]:
        sql = """
        SELECT news_id::TEXT, headline::TEXT, summary::TEXT
        FROM nlp_service.theme_news
        WHERE news_id = ANY(%(news_id)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"news_id": [topic.id for topic in developments]})
        return {row["news_id"]: f"{row['headline']}:\n{row['summary']}" for row in rows}

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        sql = """
        SELECT news_id::TEXT, url, domain_url, headline, published_at
        FROM nlp_service.theme_news tn
        JOIN nlp_service.news_sources ns ON ns.source_id = tn.source_id
        WHERE tn.news_id = ANY(%(news_ids)s)
        """
        params = {"news_ids": [text.source_text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return [
            NewsArticleCitationOutput(
                id=row["news_id"],
                name=row["domain_url"],
                link=row["url"],
                summary=row["headline"],
                last_updated_at=row["published_at"],
            )
            for row in rows
        ]


# Parent class that is not intended to be used on its own, should always use one of the child classes
@io_type
class StockEarningsText(StockText):
    id: str


@io_type
class StockEarningsSummaryText(StockEarningsText):
    text_type: ClassVar[str] = "Earnings Call Summary"

    @classmethod
    async def _get_strs_lookup(
        cls, earnings_texts: List[StockEarningsSummaryText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        sql = """
        SELECT summary_id::TEXT, summary
        FROM nlp_service.earnings_call_summaries
        WHERE summary_id = ANY(%(earnings_ids)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"earnings_ids": [summary.id for summary in earnings_texts]})
        str_lookup = {}
        for row in rows:
            output = []
            summary = row["summary"]
            for section in ["Remarks", "Questions"]:
                if section in summary and summary[section]:
                    output.append(section)
                    for point in summary[section]:
                        output.append(f"- {point['header']}: {point['detail']}")
            output_str = "\n".join(output)
            str_lookup[row["summary_id"]] = output_str
        return str_lookup


@io_type
class StockEarningsTranscriptText(StockEarningsText):
    text_type: ClassVar[str] = "Earnings Call Transcript"

    @classmethod
    async def _get_strs_lookup(
        cls, earnings_texts: List[StockEarningsSummaryText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        earnings_transcript_sql = """
            SELECT id::TEXT, transcript
            FROM company_earnings.full_earning_transcripts
            WHERE id IN %(ids)s
        """
        ch = Clickhouse()
        transcript_query_result = ch.clickhouse_client.query(
            earnings_transcript_sql,
            parameters={
                "ids": [earnings_text.id for earnings_text in earnings_texts],
            },
        )
        str_lookup = {row[0]: row[1] for row in transcript_query_result.result_rows}
        return str_lookup


@io_type
class StockEarningsSummaryPointText(StockText):
    """
    A subclass from `StockEarningsSummaryText` that only stores a point in the summary
    """

    id: int  # hash((summary_id, summary_type, summary_idx))
    text_type: ClassVar[str] = "Earnings Call Summary Point"

    summary_id: str  # UUID in DB
    summary_type: str  # "Remarks" or "Questions"
    summary_idx: int  # index of the point in the summary

    @classmethod
    async def _get_strs_lookup(cls, earnings_summary_points: List[Self]) -> Dict[TextIDType, str]:
        sql = """
            SELECT summary_id::TEXT, summary
            FROM nlp_service.earnings_call_summaries
            WHERE summary_id = ANY(%(summary_ids)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        summary_ids = list({point.summary_id for point in earnings_summary_points})
        rows = db.generic_read(sql, {"summary_ids": summary_ids})
        summary_id_to_summary = {row["summary_id"]: row["summary"] for row in rows}

        str_lookup = {}
        for point in earnings_summary_points:
            summary_per_type = summary_id_to_summary[point.summary_id][point.summary_type]
            if point.summary_idx < len(summary_per_type):
                text = summary_per_type[point.summary_idx]
                str_lookup[point.id] = f"{text['header']}: {text['detail']}"

        return str_lookup  # type: ignore

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        sql = """
        SELECT ecs.summary_id::TEXT, ms.symbol, ecs.year, ecs.quarter, ecs.created_timestamp
        FROM nlp_service.earnings_call_summaries ecs
        JOIN master_security ms ON ecs.gbi_id = ms.gbi_security_id
        WHERE summary_id = ANY(%(earnings_ids)s)
        """
        text_list: List[Self] = cast(List[Self], [text.source_text for text in texts])
        summary_id_text_map = {text.summary_id: text for text in text_list}
        params = {"earnings_ids": [text.summary_id for text in text_list]}
        str_lookup = await cls._get_strs_lookup(text_list)  # type: ignore
        rows = await db.generic_read(sql, params)
        output = []
        for row in rows:
            text = summary_id_text_map.get(row["summary_id"])
            citation = TextCitationOutput(
                # e.g. "NVDA Earnings Call - Q1 2024"
                name=f"{row['symbol'] or ''} Earnings Call - Q{row['quarter']} {row['year']}",
                published_at=row["created_timestamp"],
            )
            if text:
                point_str = str_lookup.get(text.id)
                if point_str:
                    citation.summary = f"{text.summary_type}: {point_str}"
            output.append(citation)
        return output

    @classmethod
    async def init_from_earnings_texts(
        cls, earnings_summaries: List[StockEarningsText]
    ) -> List[Self]:
        from agent_service.utils.postgres import get_psql

        sql = """
            SELECT summary_id::TEXT, summary
            FROM nlp_service.earnings_call_summaries
            WHERE summary_id = ANY(%(summary_ids)s)
        """
        db = get_psql()
        summary_id_to_summary = {s.id: s for s in earnings_summaries}
        rows = db.generic_read(sql, {"summary_ids": list(summary_id_to_summary.keys())})

        points = []
        for row in rows:
            summary_obj = summary_id_to_summary[row["summary_id"]]
            summary = row["summary"]
            for section in ["Remarks", "Questions"]:
                if section in summary and summary[section]:
                    for idx, point in enumerate(summary[section]):
                        points.append(
                            cls(
                                id=hash((row["summary_id"], section, idx)),
                                summary_id=row["summary_id"],
                                summary_type=section,
                                summary_idx=idx,
                                stock_id=summary_obj.stock_id,
                            )
                        )

        return points


@io_type
class StockHypothesisEarningsSummaryPointText(StockEarningsSummaryPointText):
    """
    A subclass from `StockEarningsSummaryPointText`
    """

    text_type: ClassVar[str] = "Hypothesis Earnings Call Summary Point"

    support_score: Score
    reason: str

    @classmethod
    async def _get_strs_lookup(
        cls, earnings_summary_points: List[StockHypothesisEarningsSummaryPointText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        return await StockEarningsSummaryPointText._get_strs_lookup(earnings_summary_points)  # type: ignore  #noqa


@io_type
class StockDescriptionText(StockText):
    id: int  # gbi_id
    text_type: ClassVar[str] = "Company Description"

    @classmethod
    async def _get_strs_lookup(
        cls, company_descriptions: List[StockDescriptionText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        sql = """
        SELECT ssm.gbi_id, cds.company_description_short
        FROM spiq_security_mapping ssm
        JOIN nlp_service.company_descriptions_short cds
        ON cds.spiq_company_id = ssm.spiq_company_id
        WHERE ssm.gbi_id = ANY(%(stocks)s)
        """
        from agent_service.utils.postgres import get_psql

        stocks = [desc.id for desc in company_descriptions]

        # get short first since we always have that

        db = get_psql()
        rows = db.generic_read(sql, {"stocks": stocks})

        descriptions = {row["gbi_id"]: row["company_description_short"] for row in rows}

        # replace with long if it exists

        long_sql = sql.replace("_short", "")

        rows = db.generic_read(long_sql, {"stocks": stocks})

        for row in rows:
            descriptions[row["gbi_id"]] = row["company_description"]

        # For some reason SPIQ includes invalid characters for apostraphes. For
        # now just replace them here, ideally a data ingestion problem to fix.
        for gbi, desc in descriptions.items():
            descriptions[gbi] = desc.replace("\x92", "'")

        return descriptions

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        return [TextCitationOutput(name=text.source_text.text_type) for text in texts]


@io_type
class StockSecFilingText(StockText):
    """
    The ID field is a serialized JSON object containing the SEC filing information.
    After deserialization, it will be a dictionary like below:
    {
        "id": "6c0b495bc8a221cf32387c0123aeee5a",
        "accessionNo": "0000320193-24-000069",
        "cik": "320193",
        "ticker": "AAPL",
        "companyName": "Apple Inc.",
        "companyNameLong": "Apple Inc. (Filer)",
        "formType": "10-Q",
        "description": "Form 10-Q - Quarterly report [Sections 13 or 15(d)]",
        "filedAt": "2024-05-02T18:04:25-04:00",
        "linkToTxt": "...",
        "linkToHtml": "...",
        "linkToXbrl": "",
        "linkToFilingDetails": "...",
        ...
    }
    """

    id: str

    text_type: ClassVar[str] = "SEC filing"

    db_id: Optional[str] = None

    @classmethod
    async def _get_strs_lookup(
        cls, sec_filing_list: List[StockSecFilingText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        # Get the available content from DB first
        output: Dict[TextIDType, str] = {}

        db_id_to_text_id = {filing.db_id: filing.id for filing in sec_filing_list if filing.db_id}
        output.update(SecFiling.get_concat_10k_10q_sections_from_db(db_id_to_text_id))  # type: ignore

        # Get the rest from SEC API directly
        filing_gbi_pairs = [
            (filing.id, filing.stock_id.gbi_id)
            for filing in sec_filing_list
            if not filing.db_id and filing.stock_id
        ]
        output.update(
            SecFiling.get_concat_10k_10q_sections_from_api(filing_gbi_pairs, insert_to_db=True)  # type: ignore
        )

        return output

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        output = []
        for text in texts:
            hl_start, hl_end = None, None
            if text.citation_snippet_context and text.citation_snippet:
                hl_start, hl_end = CompanyFilingCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=text.citation_snippet, context=text.citation_snippet_context
                )
            output.append(
                CompanyFilingCitationOutput(
                    name=text.source_text.text_type,
                    cited_snippet=text.citation_snippet_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                    inline_offset=text.citation_text_offset,
                )
            )
        return output


@io_type
class StockSecFilingSectionText(StockText):
    """
    This class is actually a "section" of `StockSecFilingText`. Basically we will split the 2 sections
    (management, risk_factors) into even smaller sections and store them in this class.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))

    text_type: ClassVar[str] = "SEC filing Section"

    filing_id: str
    header: str
    db_id: Optional[str] = None

    @classmethod
    async def init_from_filings(
        cls, filings: List[StockSecFilingText]
    ) -> List[StockSecFilingSectionText]:
        filing_texts = await StockSecFilingText._get_strs_lookup(filings)

        sections = []
        for filing in filings:
            if filing.id not in filing_texts:
                continue

            split_sections = SecFiling.split_10k_10q_into_smaller_sections(filing_texts[filing.id])
            for header, content in split_sections.items():
                sections.append(
                    cls(
                        val=content,
                        filing_id=filing.id,
                        header=header,
                        stock_id=filing.stock_id,
                        db_id=filing.db_id,
                    )
                )

        return sections

    @classmethod
    async def _get_strs_lookup(
        cls, sections: List[StockSecFilingSectionText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        # In the DB we only store `header`, not `content` to save space
        filing_text_objs = {}
        for section in sections:
            if section.filing_id not in filing_text_objs:
                filing_text_objs[section.filing_id] = StockSecFilingText(
                    id=section.filing_id, stock_id=section.stock_id, db_id=section.db_id
                )

        filing_texts = await StockSecFilingText._get_strs_lookup(list(filing_text_objs.values()))

        header_to_section = {section.header: section for section in sections}

        outputs = {}
        for filing_text in filing_texts.values():
            split_sections = SecFiling.split_10k_10q_into_smaller_sections(filing_text)
            for header, content in split_sections.items():
                if header in header_to_section:
                    outputs[header_to_section[header].id] = f"{header}: {content}"

        return outputs  # type: ignore

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        output = []
        filing_id_set = set()
        for cit in texts:
            source_text = cit.source_text
            source_text = cast(Self, cit.source_text)
            if source_text.filing_id in filing_id_set:
                continue
            filing_id_set.add(source_text.filing_id)
            hl_start, hl_end = None, None
            if cit.citation_snippet_context and cit.citation_snippet:
                hl_start, hl_end = CompanyFilingCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=cit.citation_snippet, context=cit.citation_snippet_context
                )
            output.append(
                CompanyFilingCitationOutput(
                    name="SEC filing Section",
                    cited_snippet=cit.citation_snippet_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                    inline_offset=cit.citation_text_offset,
                )
            )

        return output

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return f"{self.header}: {self.val}"


@io_type
class StockOtherSecFilingText(StockText):
    """
    Unlike `SecFilingText`, this class is used to get other types of SEC filings and the helper
    method `get_strs_lookup` is used to download the full content of the filing instead of only
    extracting a few sections.
    TODO: We may later merge this class with `SecFilingText` (and also extract certain sections
    for other types)
    """

    id: str  # SEC filing info

    text_type: ClassVar[str] = "SEC filing"

    db_id: Optional[str] = None

    @classmethod
    async def _get_strs_lookup(
        cls, sec_filing_list: List[StockOtherSecFilingText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        # Get the available content from DB first
        output: Dict[TextIDType, str] = {}

        db_id_to_text_id = {filing.db_id: filing.id for filing in sec_filing_list if filing.db_id}
        output.update(SecFiling.get_filings_content_from_db(db_id_to_text_id))  # type: ignore

        # Get the rest from SEC API directly
        filing_gbi_pairs = [
            (filing.id, filing.stock_id.gbi_id)
            for filing in sec_filing_list
            if not filing.db_id and filing.stock_id
        ]
        output.update(
            SecFiling.get_filings_content_from_api(filing_gbi_pairs, insert_to_db=True)  # type: ignore
        )

        return output

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        # the downloaded filings is very messy and not human readable
        # return a link to the filing on SEC website instead
        filing_dict = json.loads(self.id)
        ticker = filing_dict["ticker"]
        form_type = filing_dict["formType"]
        url = filing_dict[LINK_TO_FILING_DETAILS]
        val = f"See {ticker}'s {form_type} filings on [SEC official website]({url})"  # markdown
        return TextOutput(output_type=OutputType.TEXT, val=val, title=title)

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        output = []
        for text in texts:
            hl_start, hl_end = None, None
            if text.citation_snippet_context and text.citation_snippet:
                hl_start, hl_end = CompanyFilingCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=text.citation_snippet, context=text.citation_snippet_context
                )
            output.append(
                CompanyFilingCitationOutput(
                    name=text.source_text.text_type,
                    cited_snippet=text.citation_snippet_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                    inline_offset=text.citation_text_offset,
                )
            )
        return output


@io_type
class KPIText(Text):
    pid: Optional[int] = None
    explanation: Optional[str] = None  # To be used as a way to store why a KPI has been selected
    text_type: ClassVar[str] = "KPI"

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Sequence[CitationOutput]:
        output = []
        for cit in texts:
            text = cast(Self, cit.source_text)
            output.append(
                TextCitationOutput(
                    id=str(text.id),
                    name=text.val,
                    summary=text.explanation,
                )
            )
        return output


# These are not actual Text types, but build on top of them


@io_type
class TextGroup(ComplexIOBase):
    val: List[Text]

    @staticmethod
    def join(group1: TextGroup, group2: TextGroup) -> TextGroup:
        return TextGroup(val=group1.val + group2.val)

    def convert_to_str(self, id_to_str: Dict[TextIDType, str], numbering: bool = False) -> str:
        return "\n***\n".join(
            [
                (f"Text Number: {i}\n" if numbering else "") + id_to_str[text.id]
                for i, text in enumerate(self.val)
                if text.id in id_to_str
            ]
        )

    def get_citations(self, citation_ids: List[int]) -> List[Citation]:
        # do int(i) just in case GPT screwed up
        return [
            TextCitation(source_text=self.val[int(i)])
            for i in citation_ids
            if 0 <= int(i) < len(self.val)
        ]

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        from agent_service.utils.output_utils.output_construction import (
            get_output_from_io_type,
        )

        # construct a lookup for all child texts
        strings = await Text.get_all_strs(self.val)
        # TODO fix this implementation?
        return await get_output_from_io_type(strings, pg=pg, title=title)


@io_type
class EquivalentKPITexts(TextGroup):
    val: List[KPIText]  # type: ignore
    general_kpi_name: str


@io_type
class TopicProfiles(TextGroup):
    val: List[ProfileText]  # type: ignore
    topic: str


class TextOutput(Output):
    output_type: Literal[OutputType.TEXT] = OutputType.TEXT
    val: str
    # Optional top level score for this segment of text
    score: Optional[ScoreOutput] = None

    def convert_inline_citations_to_output_format(self) -> str:
        """
        Given a string representing this text's value, go through all the
        citations for this object and insert them into the text.
        """
        template = """ ```{{ "type": "citation", "citation_id": "{cit_id}" }}``` """
        citation_offset_map = {}
        char_list = list(self.val)
        for cit in self.citations:
            if not cit.inline_offset:
                continue
            citation_offset_map[cit.inline_offset] = template.format(cit_id=cit.id)

        output = []
        for i, char in enumerate(char_list):
            output.append(char)
            if i in citation_offset_map:
                output.append(citation_offset_map[i])

        return "".join(output)

    def model_post_init(self, __context: Any) -> None:
        self.val = self.convert_inline_citations_to_output_format()

    def __str__(self) -> str:
        return self.val


@io_type
class TextCitation(Citation):
    source_text: Text
    # Offset into the parent text for where this citation applies and should be inserted
    citation_text_offset: Optional[int] = None
    citation_snippet: Optional[str] = None
    # Context around the snippet that contains the snippet itself.
    citation_snippet_context: Optional[str] = None

    def __hash__(self) -> int:
        return hash(
            (
                self.source_text.__hash__(),
                self.citation_text_offset,
                self.citation_snippet,
                self.citation_snippet_context,
            )
        )

    def __eq__(self, other: Any) -> bool:
        if (
            self.source_text,
            self.citation_text_offset,
            self.citation_snippet,
            self.citation_snippet_context,
        ) == (
            other.source_text,
            other.citation_text_offset,
            other.citation_snippet,
            other.citation_snippet_context,
        ):
            return True
        return False

    @classmethod
    async def resolve_citations(cls, citations: List[Self], db: BoostedPG) -> List[CitationOutput]:
        # First group citations based on type of the source text.
        text_type_map = defaultdict(list)
        for cit in citations:
            text_type_map[type(cit.source_text)].append(cit)

        tasks = [
            typ.get_citations_for_output(text_list, db) for typ, text_list in text_type_map.items()  # type: ignore
        ]
        # List of lists, where each nested list has outputs for each type
        outputs_nested: List[Sequence[CitationOutput]] = await gather_with_concurrency(tasks)
        # Unnest into a single list
        outputs = list(chain(*outputs_nested))

        return outputs
