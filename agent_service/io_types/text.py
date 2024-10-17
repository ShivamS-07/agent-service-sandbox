from __future__ import annotations

import datetime
import json
import logging
from collections import defaultdict
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
from uuid import uuid4

import mdutils
from pydantic import Field, field_serializer
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
    DocumentCitationOutput,
    EarningsSummaryCitationOutput,
    EarningsTranscriptCitationOutput,
    KPICitationOutput,
    NewsArticleCitationOutput,
    NewsDevelopmentCitationOutput,
    ThemeCitationOutput,
    WebCitationOutput,
)
from agent_service.io_types.output import Output, OutputType
from agent_service.io_types.stock import StockID
from agent_service.io_types.text_objects import CitationTextObject, TextObject
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.date_utils import parse_date_str_in_utc
from agent_service.utils.earnings.earnings_util import (
    get_transcript_partitions_from_db,
    get_transcript_sections_from_partitions,
    insert_transcript_partitions_to_db,
    split_transcript_into_smaller_sections,
)
from agent_service.utils.event_logging import log_event
from agent_service.utils.sec.constants import LINK_TO_FILING_DETAILS
from agent_service.utils.sec.sec_api import SecFiling
from agent_service.utils.string_utils import get_sections

logger = logging.getLogger(__name__)

TextIDType = Union[str, int]

DEFAULT_TEXT_TYPE = "Misc"


@io_type
class Text(ComplexIOBase):
    id: TextIDType = Field(default_factory=lambda: str(uuid4()))
    val: str = ""
    text_type: ClassVar[str] = DEFAULT_TEXT_TYPE
    stock_id: Optional[StockID] = None
    timestamp: Optional[datetime.datetime] = None
    title: Optional[str] = None  # useful for lists of Texts to be displayed in a single widget
    # Text objects are rich "widgets" that are rendered on the UI. For example,
    # inline citations, stock links, portfolio links, etc. could all be text
    # objects.
    text_objects: List[TextObject] = Field(default_factory=list)

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Text):
            return self.id == other.id
        return False

    def to_citation_title(self) -> str:
        return self.text_type

    def reset_id(self) -> None:
        pass

    @staticmethod
    async def multi_text_rich_output(pg: BoostedPG, texts: List[Text]) -> List[TextOutput]:
        all_citations = []
        citation_text_map = {}
        for text in texts:
            all_citations_for_text = text.get_all_citations()
            all_citations.extend(all_citations_for_text)
            for cit in all_citations_for_text:
                citation_text_map[cit] = text

        # Maps Citation to List[CitationOutput]
        mapped_citations = await Citation.resolve_all_citations_mapped(
            citations=all_citations, db=pg
        )

        # Maps Text to List[CitationOutput]
        text_to_output_citations: Dict[Text, List[CitationOutput]] = {text: [] for text in texts}
        for citation, output_citations in mapped_citations.items():
            text = citation_text_map[citation]
            text_to_output_citations[text].extend(output_citations)

        tasks = [
            text.to_rich_output(pg=pg, cached_resolved_citations=text_to_output_citations.get(text))
            for text in texts
        ]
        return await gather_with_concurrency(tasks)

    async def to_rich_output(
        self,
        pg: BoostedPG,
        title: str = "",
        cached_resolved_citations: Optional[List[CitationOutput]] = None,
    ) -> Output:
        # Get the citations for the current Text object, as well as any
        # citations from "child" (input) texts.
        if cached_resolved_citations is None:
            tasks = [
                Citation.resolve_all_citations(citations=self.get_all_citations(), db=pg),
                self.get_citations_for_output(texts=[TextCitation(source_text=self)], db=pg),
            ]
            outputs = await gather_with_concurrency(tasks)
            citations = outputs[0] + list(outputs[1].values())
        else:
            citations = cached_resolved_citations
        text = await self.get()

        output_val = TextOutput(
            val=text.val,
            title=title,
            citations=citations,
            score=ScoreOutput.from_entry_list(self.history),
        )

        citation_text_objects = []
        for cit in citations:
            if not cit.inline_offset:
                continue
            citation_text_objects.append(
                CitationTextObject(index=cit.inline_offset, citation_id=cit.id)
            )

        output_val.render_text_objects(
            text_objects=(self.text_objects or []) + citation_text_objects  # type: ignore
        )
        return output_val

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
        include_symbols: bool = False,
        include_timestamps: bool = True,  # only actually used if include_header is True
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
        if include_header and include_timestamps:
            timestamp_lookup = {}
            for cat_texts in categories.values():
                for text in cat_texts:
                    timestamp_lookup[text.id] = text.timestamp
        else:
            timestamp_lookup = None
        if include_header and include_symbols:
            symbol_lookup = {}
            for cat_texts in categories.values():
                for text in cat_texts:
                    if (
                        isinstance(text, StockText)
                        and text.stock_id is not None
                        and text.stock_id.symbol is not None
                    ):
                        symbol_lookup[text.id] = text.stock_id.symbol
        else:
            symbol_lookup = None
        strs_lookup: Dict[TextIDType, str] = {}
        # For every subclass of Text, fetch data
        lookups: List[Dict[TextIDType, str]] = await gather_with_concurrency(
            [
                textclass.get_strs_lookup(
                    texts,
                    include_header=include_header,
                    timestamp_lookup=timestamp_lookup,
                    symbol_lookup=symbol_lookup,
                )
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
                if id_rep not in strs_lookup:
                    logger.error(f"Text ID not found for type {cls}!: {id_rep=}")
                    log_event(
                        event_name="agent-svc-text-id-lookup-keyerror",
                        event_data={"text_type": str(cls), "missing_id": id_rep},
                    )
                    return "Text not found"
                return strs_lookup[id_rep]

        return convert_ids_to_strs(strs_lookup, texts_as_ids)

    @classmethod
    async def get_strs_lookup(
        cls,
        texts: List[Self],
        include_header: bool = False,
        timestamp_lookup: Optional[Dict[TextIDType, Optional[datetime.datetime]]] = None,
        symbol_lookup: Optional[Dict[TextIDType, str]] = None,
    ) -> Dict[TextIDType, str]:
        strs_lookup = await cls._get_strs_lookup(texts)
        if include_header:
            for id, val in strs_lookup.items():
                if timestamp_lookup and id in timestamp_lookup and timestamp_lookup[id] is not None:
                    date_str = f"Date: {timestamp_lookup[id].date()}\n"  # type: ignore
                else:
                    date_str = ""
                if symbol_lookup and id in symbol_lookup:
                    symbol_str = f"Stock: {symbol_lookup[id]}\n"
                else:
                    symbol_str = ""
                strs_lookup[id] = f"Text type: {cls.text_type}\n{symbol_str}{date_str}Text:\n{val}"
        return strs_lookup

    @classmethod
    async def _get_strs_lookup(cls, texts: List[Self]) -> Dict[TextIDType, str]:
        return {text.id: text.val for text in texts}

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        """
        Given a list of text citations of this type, return a mapping from the
        input citation to a list of output citations.
        """
        return {}

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

    def to_citation_title(self) -> str:
        title = super().to_citation_title()
        if self.stock_id:
            stock_name = self.stock_id.symbol or self.stock_id.company_name
            title = f"{stock_name} {title}"

        return title


@io_type
class StatisticsText(Text):
    id: str
    text_type: ClassVar[str] = "STATISTICS"


@io_type
class EarningsPeersText(StockText):
    affecting_stock_id: Optional[StockID] = None
    year: int
    quarter: int


@io_type
class StockNewsDevelopmentText(NewsText, StockText):
    id: str
    text_type: ClassVar[str] = "News Development Summary"

    # If this is non-null, don't show the topic sumamry to GPT, just show
    # summaries of the most recent articles since this date. This can be useful
    # if the topic is brand new or if the user wants info on a very specific
    # time range.
    only_get_articles_start: Optional[datetime.datetime] = None
    only_get_articles_end: Optional[datetime.datetime] = None

    @classmethod
    async def _get_strs_lookup(
        cls,
        news_topics: List[StockNewsDevelopmentText],
    ) -> Dict[TextIDType, str]:  # type: ignore
        news_topics_articles_only = {
            topic.id: (topic.only_get_articles_start, topic.only_get_articles_end)
            for topic in news_topics
            if topic.only_get_articles_start and topic.only_get_articles_end
        }
        news_topics = [topic for topic in news_topics if not topic.only_get_articles_start]
        article_sql = """
        SELECT snt.topic_id::TEXT, (topic_descriptions->-1->>0)::TEXT AS description,
               snt.topic_label,
               json_agg(json_build_object('headline', sn.headline,
                                     'summary', sn.summary,
                                     'published_at', sn.published_at,
                                     'is_top_source', sn.is_top_source))
          AS news_items
        FROM nlp_service.stock_news_topics snt
        JOIN nlp_service.stock_news sn ON snt.topic_id = sn.topic_id
        WHERE snt.topic_id = ANY(%(topic_ids)s)
        GROUP BY snt.topic_id
        """

        sql = """
        SELECT topic_id::TEXT, topic_label, (topic_descriptions->-1->>0)::TEXT AS description
        FROM nlp_service.stock_news_topics
        WHERE topic_id = ANY(%(topic_ids)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"topic_ids": [topic.id for topic in news_topics]})
        rows_with_articles = []
        if news_topics_articles_only:
            rows_with_articles = db.generic_read(
                article_sql, {"topic_ids": list(news_topics_articles_only.keys())}
            )
        for row in rows_with_articles:
            # For the rows with articles, create a description based on the
            # article headlines and summaries.
            news_items = [
                item
                for item in row["news_items"]
                # Make sure the published_at date is between the start and end
                # dates. news_topics_articles_only stores tuples of (start, end)
                # pairs for each topic.
                if news_topics_articles_only[row["topic_id"]][0]
                <= parse_date_str_in_utc(item["published_at"])
                <= news_topics_articles_only[row["topic_id"]][1]
            ]
            news_items = list(
                sorted(
                    news_items, key=lambda n: (n["is_top_source"], n["published_at"]), reverse=True
                )
            )
            if news_items:
                best_article = news_items[0]
                row["description"] = best_article["summary"]
                row["topic_label"] = best_article["headline"]
        return {
            row["topic_id"]: f"{row['topic_label']}: {row['description']}"
            for row in rows + rows_with_articles
        }

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        text_id_map = {text.source_text.id: text for text in texts}
        sql = """
        SELECT snt.topic_id::TEXT, topic_label, (topic_descriptions->-1->>0)::TEXT AS summary,
               COUNT(sn.news_id) AS num_articles, MAX(sn.published_at) AS last_updated
        FROM nlp_service.stock_news_topics snt
        JOIN nlp_service.stock_news sn ON sn.topic_id = snt.topic_id
        WHERE sn.topic_id = ANY(%(topic_ids)s)
        GROUP BY snt.topic_id
        """
        params = {"topic_ids": [text.source_text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return {
            text_id_map[row["topic_id"]]: [
                NewsDevelopmentCitationOutput(
                    internal_id=row["topic_id"],
                    name=row["topic_label"],
                    summary=row["summary"],
                    last_updated_at=row["last_updated"],
                    num_articles=row["num_articles"],
                    inline_offset=text_id_map[row["topic_id"]].citation_text_offset,
                )
            ]
            for row in rows
        }


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
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        text_id_map = {text.source_text.id: text for text in texts}
        sql = """
        SELECT news_id::TEXT, url, domain_url, headline, published_at
        FROM nlp_service.stock_news sn
        JOIN nlp_service.news_sources ns ON ns.source_id = sn.source_id
        WHERE sn.news_id = ANY(%(news_ids)s)
        """
        params = {"news_ids": [text.source_text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return {
            text_id_map[row["news_id"]]: [
                NewsArticleCitationOutput(
                    internal_id=row["news_id"],
                    name=row["domain_url"],
                    link=row["url"],
                    summary=row["headline"],
                    last_updated_at=row["published_at"],
                    inline_offset=text_id_map[row["news_id"]].citation_text_offset,
                )
            ]
            for row in rows
        }


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
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        text_id_map = {text.source_text.id: text for text in texts}
        sql = """
        SELECT news_id::TEXT, url, domain_url, headline, published_at
        FROM nlp_service.news_pool np
        JOIN nlp_service.news_sources ns ON ns.source_id = np.source_id
        WHERE np.news_id = ANY(%(news_ids)s)
        """
        params = {"news_ids": [text.source_text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return {
            text_id_map[row["news_id"]]: [
                NewsArticleCitationOutput(
                    internal_id=row["news_id"],
                    name=row["domain_url"],
                    link=row["url"],
                    summary=row["headline"],
                    last_updated_at=row["published_at"],
                    inline_offset=text_id_map[row["news_id"]].citation_text_offset,
                )
            ]
            for row in rows
        }


@io_type
class CustomDocumentSummaryText(StockText):
    id: str
    requesting_user: str
    text_type: ClassVar[str] = "User Document Summary"

    @field_serializer("val")
    def serialize_val(self, val: str, _info: Any) -> str:
        # Make sure we don't serialize unnecessary data, we only want to serialize the ID.
        return ""

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
                texts[id] = f"{chunk_info.headline}:\n{chunk_info.long_summary}"
        return texts

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        # group by user id - we are assuming that anyone who is authed for the agent
        # has priv to see any documents utilized by the agent.
        text_citation_by_user: Dict[str, List[TextCitation]] = defaultdict(list)
        text_id_map = {text.source_text.id: text for text in texts}
        for cit in texts:
            text_citation_by_user[cit.source_text.requesting_user].append(cit)  # type: ignore

        output_citations: Dict[TextCitation, List[CitationOutput]] = defaultdict(list)
        # TODO this will need to change once custom docs have better citation
        # support. Currently we just cite ALL chunks in the file.
        for user, citations in text_citation_by_user.items():
            article_info = await get_custom_doc_articles_info(
                user, [str(cit.source_text.id) for cit in citations]
            )
            for chunk_id, chunk_info in dict(article_info.file_chunk_info).items():
                file_paths = list(chunk_info.file_paths)
                if len(file_paths) == 0:
                    citation_name = chunk_info.file_id
                else:
                    # pick an arbitrary custom file path
                    citation_name = file_paths[0].lstrip("/")

                chunk_cit = text_id_map.get(chunk_info.chunk_id)
                if not chunk_cit:
                    continue
                for citation in chunk_info.citations:
                    hl_start, hl_end = None, None
                    if citation.citation_snippet and citation.citation_context:
                        hl_start, hl_end = DocumentCitationOutput.get_offsets_from_snippets(
                            smaller_snippet=citation.citation_snippet,
                            context=citation.citation_context,
                        )
                    output_citations[chunk_cit].append(
                        CustomDocumentCitationOutput(
                            id=citation.citation_id,
                            internal_id=citation.citation_id,
                            chunk_id=chunk_id,
                            name=f"User Document: {citation_name}",
                            last_updated_at=chunk_info.upload_time.ToDatetime(),
                            custom_doc_id=chunk_info.file_id,
                            inline_offset=chunk_cit.citation_text_offset,
                            summary=citation.citation_context,
                            snippet_highlight_start=hl_start,
                            snippet_highlight_end=hl_end,
                        )
                    )
                if not chunk_info.citations:
                    # if this happens it implies the custom doc is not properly processed
                    logger.warning("No citations found for custom doc chunk %s", chunk_id)
                    output_citations[chunk_cit].append(
                        CustomDocumentCitationOutput(
                            chunk_id=chunk_id,
                            name=f"User Document: {citation_name}",
                            last_updated_at=chunk_info.upload_time.ToDatetime(),
                            custom_doc_id=chunk_info.file_id,
                            inline_offset=chunk_cit.citation_text_offset,
                        )
                    )
        return output_citations  # type: ignore


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
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        text_id_map = {text.source_text.id: text for text in texts}
        sql = """
        SELECT theme_id::TEXT, theme_name::TEXT AS name, theme_description, last_modified
        FROM nlp_service.themes
        WHERE theme_id = ANY(%(theme_id)s)
        """
        rows = await db.generic_read(sql, {"theme_id": [text.source_text.id for text in texts]})
        return {
            text_id_map[row["theme_id"]]: [
                ThemeCitationOutput(
                    internal_id=row["theme_id"],
                    name="Theme: " + row["name"],
                    summary=row["theme_description"],
                    last_updated_at=row["last_modified"],
                    inline_offset=text_id_map[row["theme_id"]].citation_text_offset,
                )
            ]
            for row in rows
        }


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
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        text_id_map = {text.source_text.id: text for text in texts}
        sql = """
        SELECT development_id::TEXT, label::TEXT, description, development_time
        FROM nlp_service.theme_developments
        WHERE development_id = ANY(%(development_id)s)
        """
        rows = await db.generic_read(
            sql, {"development_id": [text.source_text.id for text in texts]}
        )
        return {
            text_id_map[row["development_id"]]: [
                NewsDevelopmentCitationOutput(
                    internal_id=row["development_id"],
                    name="News Development: " + row["label"],
                    summary=row["description"],
                    last_updated_at=row["development_time"],
                    inline_offset=text_id_map[row["development_id"]].citation_text_offset,
                )
            ]
            for row in rows
        }


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
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        text_id_map = {text.source_text.id: text for text in texts}
        sql = """
        SELECT news_id::TEXT, url, domain_url, headline, published_at
        FROM nlp_service.theme_news tn
        JOIN nlp_service.news_sources ns ON ns.source_id = tn.source_id
        WHERE tn.news_id = ANY(%(news_ids)s)
        """
        params = {"news_ids": [text.source_text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return {
            text_id_map[row["news_id"]]: [
                NewsArticleCitationOutput(
                    internal_id=row["news_id"],
                    name=row["domain_url"],
                    link=row["url"],
                    summary=row["headline"],
                    last_updated_at=row["published_at"],
                    inline_offset=text_id_map[row["news_id"]].citation_text_offset,
                )
            ]
            for row in rows
        }


# Parent class that is not intended to be used on its own, should always use one of the child classes
@io_type
class StockEarningsText(StockText):
    year: Optional[int] = None
    quarter: Optional[int] = None

    def to_citation_title(self) -> str:
        parts = []
        if self.stock_id:
            parts.append(self.stock_id.symbol or self.stock_id.company_name)
            parts.append(" ")
        parts.append("Earnings Call")
        parts_in_parens = []
        if self.year and self.quarter:
            year_num = str(self.year)[2:]
            parts_in_parens.append(f"Q{self.quarter}'{year_num}")
        if self.timestamp:
            ts_str = self.timestamp.strftime("%Y-%m-%d")
            parts_in_parens.append(f"{ts_str}")
        if parts_in_parens:
            joined = " - ".join(parts_in_parens)
            parts.append(f" ({joined})")
        # Result should look like this:
        # AAPL Earnings Call (Q1'25 - May 22, 2024)
        return "".join(parts)


@io_type
class StockEarningsSummaryText(StockEarningsText):
    text_type: ClassVar[str] = "Earnings Call"

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

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        """
        We do something different here: Originally each object is a point in the summary, with the
        content of '<header>: <details>'. And we show the full text, and highlight the sentences GPT
        thinks are relevant.
        Now, we switch to the precomputed snippets. Each point can have one or multiple paragraphs
        and highlighted sentences. So we break down the point into multiple outputs, each output
        corresponds to a paragraph.
        """

        summary_id_to_text: Dict[str, List[TextCitation]] = defaultdict(list)
        for text_obj in texts:
            summary_id_to_text[text_obj.source_text.id].append(text_obj)  # type: ignore

        sql = """
            SELECT summary_id::TEXT, gbi_id, summary, year, quarter, created_timestamp
            FROM nlp_service.earnings_call_summaries
            WHERE summary_id = ANY(%(summary_ids)s)
        """
        rows = await db.generic_read(sql, {"summary_ids": list(summary_id_to_text.keys())})
        summary_id_to_row = {row["summary_id"]: row for row in rows}

        outputs: Dict[TextCitation, List[CitationOutput]] = defaultdict(list)
        for summary_id, text_citation_list in summary_id_to_text.items():
            row = summary_id_to_row.get(summary_id)
            if not row:
                continue
            for text_citation in text_citation_list:
                summary_dict: Dict = row["summary"]

                text: Self = text_citation.source_text  # type: ignore

                stock = text.stock_id
                if not stock:
                    continue
                # e.g. "NVDA Earnings Call - Q1 2024"
                citation_name = text.to_citation_title()
                # we don't have transcript references filled in. Use the snippet in the object
                full_context = text_citation.citation_snippet_context
                snippet = text_citation.citation_snippet
                hl_start, hl_end = DocumentCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=snippet, context=full_context
                )
                # At any point, if we fail, just append the reference to the
                fallback_citation = EarningsSummaryCitationOutput(
                    name=citation_name,
                    last_updated_at=text_citation.source_text.timestamp,
                    inline_offset=text_citation.citation_text_offset,
                    summary=full_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                )

                if hl_start is None or hl_end is None or full_context is None or snippet is None:
                    # No highlighting, can't tie it back to the transcript.
                    outputs[text_citation].append(fallback_citation)
                    continue

                remarks = summary_dict.get("Remarks") or []
                questions = summary_dict.get("Questions") or []
                found_references = False

                for point in remarks + questions:
                    if (
                        "references" not in point
                        or "reference" not in point["references"]
                        or not point["references"]["reference"]
                    ):
                        continue
                    # If we're here, we have a point with references. "snippet" in
                    # this case is a snippet from the earnings call SUMMARY. We
                    # basically need to decide if the current point we're looking at
                    # is the one referenced, and then we need to swap out the
                    # summary citation with the transcript citation.
                    if snippet not in point["detail"] and snippet not in point["header"]:
                        continue

                    references = point["references"]["reference"]
                    found_references = True
                    for reference in references:
                        # The annoying thing is - the highlight sentences are not guranteed to be
                        # adjacent. To avoid more troubles, we just find the first and last sentence
                        # and highlight the whole thing.
                        full_context = " ".join(reference["paragraph"])
                        highlight_sentences: List[str] = reference["highlight"]
                        if not highlight_sentences:
                            continue
                        hl_start, _ = DocumentCitationOutput.get_offsets_from_snippets(
                            smaller_snippet=highlight_sentences[0], context=full_context
                        )
                        _, hl_end = DocumentCitationOutput.get_offsets_from_snippets(
                            smaller_snippet=highlight_sentences[-1], context=full_context
                        )
                        outputs[text_citation].append(
                            EarningsTranscriptCitationOutput(
                                name=citation_name,
                                internal_id=EarningsTranscriptCitationOutput.get_citation_internal_id(
                                    gbi_id=row["gbi_id"], year=row["year"], quarter=row["quarter"]
                                ),
                                last_updated_at=text_citation.source_text.timestamp,
                                inline_offset=text_citation.citation_text_offset,
                                summary=full_context,
                                snippet_highlight_start=hl_start,
                                snippet_highlight_end=hl_end,
                            )
                        )
                if not found_references:
                    outputs[text_citation].append(fallback_citation)

        return outputs  # type: ignore


@io_type
class StockEarningsTranscriptText(StockEarningsText):
    text_type: ClassVar[str] = "Earnings Call"

    @field_serializer("val")
    def serialize_val(self, val: str, _info: Any) -> str:
        # Make sure we don't serialize unnecessary data, we only want to serialize the ID.
        return ""

    @classmethod
    async def _get_strs_lookup(
        cls, earnings_texts: List[StockEarningsTranscriptText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        earnings_transcript_sql = """
            SELECT id::TEXT AS id, transcript, fiscal_year, fiscal_quarter
            FROM company_earnings.full_earning_transcripts
            WHERE id IN %(ids)s
        """
        ch = Clickhouse()
        transcript_query_result = await ch.generic_read(
            earnings_transcript_sql,
            params={
                "ids": [earnings_text.id for earnings_text in earnings_texts],
            },
        )
        str_lookup = {row["id"]: row["transcript"] for row in transcript_query_result}
        return str_lookup

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        output: Dict[TextCitation, List[CitationOutput]] = defaultdict(list)
        for text in texts:
            hl_start, hl_end = None, None
            if text.citation_snippet_context and text.citation_snippet:
                hl_start, hl_end = DocumentCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=text.citation_snippet, context=text.citation_snippet_context
                )
            output[text].append(
                EarningsTranscriptCitationOutput(
                    internal_id=str(text.source_text.id),
                    name=text.source_text.to_citation_title(),
                    summary=text.citation_snippet_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                    inline_offset=text.citation_text_offset,
                    last_updated_at=text.source_text.timestamp,
                )
            )

        return output  # type: ignore


@io_type
class StockEarningsTranscriptSectionText(StockEarningsText):
    """
    This class is actually a "section" of `StockEarningsTranscriptText`. We split the transcript into
    smaller (relatively) self-contained sections.
    """

    id: Union[int, str]  # hash((self.filing_id, self.header))  (str for backwards compatibility)

    text_type: ClassVar[str] = "Earnings Transcript Section"

    transcript_id: str
    starting_line_num: int
    ending_line_num: int

    @field_serializer("val")
    def serialize_val(self, val: str, _info: Any) -> str:
        # Make sure we don't serialize unnecessary data, we only want to serialize the ID.
        return ""

    # For identifying the same texts across runs (different hash seeds)
    def reset_id(self) -> None:
        self.id = hash((self.transcript_id, self.starting_line_num, self.ending_line_num))

    @classmethod
    async def init_from_full_text_data(
        cls,
        transcripts: List[StockEarningsTranscriptText],
        context: PlanRunContext,
        cache_new_data: bool = True,
    ) -> List[StockEarningsTranscriptSectionText]:
        transcripts_lookup = await StockEarningsTranscriptText._get_strs_lookup(transcripts)

        partition_lookup = await get_transcript_partitions_from_db(
            [str(transcript.id) for transcript in transcripts]
        )

        data_to_cache: Dict[str, List[Tuple[int, int]]] = {}

        transcript_section_texts = []
        for transcript in transcripts:
            if transcript.id not in transcripts_lookup:
                continue
            if transcripts_lookup[transcript.id] == "":
                logger.warning(f"Got empty transcript for {transcript.id}")
                continue

            partition_data_from_db = partition_lookup.get(str(transcript.id))
            if partition_data_from_db:
                transcript_partitions = get_transcript_sections_from_partitions(
                    transcripts_lookup[transcript.id], partition_data_from_db
                )
            else:
                transcript_partitions = await split_transcript_into_smaller_sections(
                    transcript_id=str(transcript.id),
                    transcript=transcripts_lookup[transcript.id],
                    context=context,
                )
                data_to_cache[str(transcript.id)] = list(transcript_partitions.keys())
            for line_num_bounds, content in transcript_partitions.items():
                starting_line_num = line_num_bounds[0]
                ending_line_num = line_num_bounds[1]
                transcript_section_texts.append(
                    cls(
                        id=hash((transcript.id, starting_line_num, ending_line_num)),
                        val=content,
                        transcript_id=str(transcript.id),
                        year=transcript.year,
                        quarter=transcript.quarter,
                        timestamp=transcript.timestamp,
                        starting_line_num=starting_line_num,
                        ending_line_num=ending_line_num,
                    )
                )

        if cache_new_data and (len(data_to_cache) == 0):
            await insert_transcript_partitions_to_db(data_to_cache)

        return transcript_section_texts

    @classmethod
    async def _get_strs_lookup(
        cls, sections: List[StockEarningsTranscriptSectionText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        transcript_ids = set([section.transcript_id for section in sections])

        earnings_transcript_sql = """
            SELECT id::TEXT AS id, transcript, fiscal_year, fiscal_quarter
            FROM company_earnings.full_earning_transcripts
            WHERE id IN %(ids)s
        """
        ch = Clickhouse()
        transcript_query_result = await ch.generic_read(
            earnings_transcript_sql,
            params={"ids": list(transcript_ids)},
        )
        transcript_lookup = {
            row["id"]: row["transcript"].split("\n") for row in transcript_query_result
        }

        outputs = {}
        for section in sections:
            transcript_lines = transcript_lookup[section.transcript_id]
            outputs[section.id] = "\n".join(
                transcript_lines[section.starting_line_num : section.ending_line_num]
            )
        return outputs  # type: ignore

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        output: Dict[TextCitation, List[CitationOutput]] = defaultdict(list)

        for text in texts:
            if isinstance(text.source_text, StockEarningsTranscriptSectionText):
                hl_start, hl_end = None, None
                if text.citation_snippet_context and text.citation_snippet:
                    hl_start, hl_end = DocumentCitationOutput.get_offsets_from_snippets(
                        smaller_snippet=text.citation_snippet, context=text.citation_snippet_context
                    )
                # Use the transcript id to map back to the full transcript
                output[text].append(
                    EarningsTranscriptCitationOutput(
                        internal_id=str(text.source_text.transcript_id),
                        name=text.source_text.to_citation_title(),
                        summary=text.citation_snippet_context,
                        snippet_highlight_start=hl_start,
                        snippet_highlight_end=hl_end,
                        inline_offset=text.citation_text_offset,
                        last_updated_at=text.source_text.timestamp,
                    )
                )
        return output  # type: ignore


@io_type
class StockEarningsSummaryPointText(StockEarningsText):
    """
    A subclass from `StockEarningsSummaryText` that only stores a point in the summary
    """

    id: int  # hash((summary_id, summary_type, summary_idx))
    text_type: ClassVar[str] = "Earnings Call Summary Point"

    summary_id: str  # UUID in DB
    summary_type: str  # "Remarks" or "Questions"
    summary_idx: int  # index of the point in the summary

    def reset_id(self) -> None:
        self.id = hash(((self.summary_id, self.summary_type, self.summary_idx)))

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
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        """
        We do something different here: Originally each object is a point in the summary, with the
        content of '<header>: <details>'. And we show the full text, and highlight the sentences GPT
        thinks are relevant.
        Now, we switch to the precomputed snippets. Each point can have one or multiple paragraphs
        and highlighted sentences. So we break down the point into multiple outputs, each output
        corresponds to a paragraph.
        """

        summary_id_to_texts: Dict[str, List[TextCitation]] = defaultdict(list)
        for text_obj in texts:
            summary_id_to_texts[text_obj.source_text.summary_id].append(text_obj)  # type: ignore

        sql = """
            SELECT summary_id::TEXT, gbi_id, summary, year, quarter, created_timestamp
            FROM nlp_service.earnings_call_summaries
            WHERE summary_id = ANY(%(summary_ids)s)
        """
        rows = await db.generic_read(sql, {"summary_ids": list(summary_id_to_texts.keys())})
        summary_id_to_row = {row["summary_id"]: row for row in rows}

        outputs: Dict[TextCitation, List[CitationOutput]] = defaultdict(list)
        for summary_id, text_citations in summary_id_to_texts.items():
            row = summary_id_to_row.get(summary_id)
            if not row:
                continue
            summary_dict: Dict = row["summary"]

            for text_citation in text_citations:
                text: Self = text_citation.source_text  # type: ignore

                stock = text.stock_id
                if not stock:
                    continue
                # e.g. "NVDA Earnings Call - Q1 2024"
                text_citation.source_text.year = row["year"]  # type: ignore
                text_citation.source_text.quarter = row["quarter"]  # type: ignore
                citation_name = text_citation.source_text.to_citation_title()

                summary_point: Dict = summary_dict[text.summary_type][text.summary_idx]
                if (
                    "references" not in summary_point
                    or "reference" not in summary_point["references"]
                    or not summary_point["references"]["reference"]
                ):
                    # we don't have transcript references filled in. Use the snippet in the object
                    full_context: str = text_citation.citation_snippet_context  # type: ignore
                    snippet: str = text_citation.citation_snippet  # type: ignore
                    hl_start, hl_end = DocumentCitationOutput.get_offsets_from_snippets(
                        smaller_snippet=snippet, context=full_context
                    )
                    outputs[text_citation].append(
                        EarningsSummaryCitationOutput(
                            name=citation_name,
                            last_updated_at=text_citation.source_text.timestamp,
                            inline_offset=text_citation.citation_text_offset,
                            summary=full_context,
                            snippet_highlight_start=hl_start,
                            snippet_highlight_end=hl_end,
                        )
                    )
                else:
                    references = summary_point["references"]["reference"]
                    for reference in references:
                        # The annoying thing is - the highlight sentences are not guranteed to be
                        # adjacent. To avoid more troubles, we just find the first and last sentence
                        # and highlight the whole thing.
                        full_context = " ".join(reference["paragraph"])
                        highlight_sentences: List[str] = reference["highlight"]
                        if not highlight_sentences:
                            continue
                        hl_start, _ = DocumentCitationOutput.get_offsets_from_snippets(
                            smaller_snippet=highlight_sentences[0], context=full_context
                        )
                        _, hl_end = DocumentCitationOutput.get_offsets_from_snippets(
                            smaller_snippet=highlight_sentences[-1], context=full_context
                        )
                        outputs[text_citation].append(
                            EarningsTranscriptCitationOutput(
                                internal_id=EarningsTranscriptCitationOutput.get_citation_internal_id(
                                    gbi_id=row["gbi_id"], year=row["year"], quarter=row["quarter"]
                                ),
                                name=citation_name,
                                last_updated_at=text_citation.source_text.timestamp,
                                inline_offset=text_citation.citation_text_offset,
                                summary=full_context,
                                snippet_highlight_start=hl_start,
                                snippet_highlight_end=hl_end,
                            )
                        )

        return outputs  # type: ignore

    @classmethod
    async def init_from_full_text_data(
        cls, earnings_summaries: List[StockEarningsSummaryText]
    ) -> List[Self]:
        from agent_service.utils.postgres import get_psql

        sql = """
            SELECT summary_id::TEXT, year, quarter,
                CASE
                    WHEN jsonb_typeof(summary->'Remarks') = 'array'
                        THEN jsonb_array_length(summary->'Remarks')
                        ELSE 0
                END AS remarks_length,
                CASE
                    WHEN jsonb_typeof(summary->'Questions') = 'array'
                        THEN jsonb_array_length(summary->'Questions')
                        ELSE 0
                END AS questions_length
            FROM nlp_service.earnings_call_summaries
            WHERE summary_id = ANY(%(summary_ids)s)
        """

        db = get_psql()
        summary_id_to_summary = {s.id: s for s in earnings_summaries}
        rows = db.generic_read(sql, {"summary_ids": list(summary_id_to_summary.keys())})

        points: List[Self] = []
        for row in rows:
            summary_id = row["summary_id"]
            summary_obj = summary_id_to_summary[summary_id]

            remarks_length = row["remarks_length"]
            questions_length = row["questions_length"]

            points.extend(
                (
                    cls(
                        id=hash((summary_id, "Remarks", idx)),
                        summary_id=summary_id,
                        summary_type="Remarks",
                        summary_idx=idx,
                        stock_id=summary_obj.stock_id,
                        timestamp=summary_obj.timestamp,
                        year=row["year"],
                        quarter=row["quarter"],
                    )
                    for idx in range(remarks_length)
                )
            )

            points.extend(
                (
                    cls(
                        id=hash((summary_id, "Questions", idx)),
                        summary_id=summary_id,
                        summary_type="Questions",
                        summary_idx=idx,
                        stock_id=summary_obj.stock_id,
                        timestamp=summary_obj.timestamp,
                        year=row["year"],
                        quarter=row["quarter"],
                    )
                    for idx in range(questions_length)
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

    @field_serializer("val")
    def serialize_val(self, val: str, _info: Any) -> str:
        # Make sure we don't serialize unnecessary data, we only want to serialize the ID.
        return ""

    @classmethod
    async def _get_strs_lookup(
        cls, company_descriptions: List[StockDescriptionText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        from agent_service.utils.postgres import get_psql

        stocks = [desc.id for desc in company_descriptions]

        db = get_psql()
        descriptions = db.get_company_descriptions(stocks)

        # For some reason SPIQ includes invalid characters for apostraphes. For
        # now just replace them here, ideally a data ingestion problem to fix.
        for gbi, desc in descriptions.items():
            if desc is not None:
                descriptions[gbi] = desc.replace("\x92", "'")
            else:
                descriptions[gbi] = "No description found"

        return descriptions  # type: ignore

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        output = defaultdict(list)
        for text in texts:
            hl_start, hl_end = None, None
            if text.citation_snippet_context and text.citation_snippet:
                hl_start, hl_end = DocumentCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=text.citation_snippet, context=text.citation_snippet_context
                )
            output[text].append(
                DocumentCitationOutput(
                    name=text.source_text.to_citation_title(),
                    summary=text.citation_snippet_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                    inline_offset=text.citation_text_offset,
                    last_updated_at=text.source_text.timestamp,
                )
            )

        return output  # type: ignore


@io_type
class StockDescriptionSectionText(StockText):
    id: Union[int, str]  # hash((self.filing_id, self.header))  (str for backwards compatibility)
    description_id: int
    header: str

    text_type: ClassVar[str] = "Company Description Section"

    @field_serializer("val")
    def serialize_val(self, val: str, _info: Any) -> str:
        # Make sure we don't serialize unnecessary data, we only want to serialize the ID.
        return ""

    @classmethod
    async def init_from_full_text_data(
        cls, descriptions: List[StockDescriptionText]
    ) -> List[StockDescriptionSectionText]:
        description_texts = await StockDescriptionText._get_strs_lookup(descriptions)

        all_sections = []
        for description in descriptions:
            if description.id not in description_texts:
                continue

            sections = get_sections(description_texts[description.id])
            for header, content in sections.items():
                all_sections.append(
                    cls(
                        id=hash((description.id, header)),
                        val=content,
                        description_id=description.id,
                        header=header,
                        stock_id=description.stock_id,
                        timestamp=description.timestamp,
                    )
                )
        return all_sections

    @classmethod
    async def _get_strs_lookup(
        cls, sections: List[StockDescriptionSectionText]
    ) -> Dict[TextIDType, str]:
        from agent_service.utils.postgres import get_psql

        description_ids = [section.description_id for section in sections]
        sections_by_desc_ids: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

        for section in sections:
            sections_by_desc_ids[section.description_id].append((int(section.id), section.header))

        db = get_psql()
        desc_lookup = db.get_company_descriptions(description_ids)

        output = {}
        for gbi_id, desc in desc_lookup.items():
            desc = desc.replace("\x92", "'")
            desc_sections = get_sections(desc)
            for section_of_interest in sections_by_desc_ids[gbi_id]:
                section_id = section_of_interest[0]
                section_header = section_of_interest[1]
                output[section_id] = desc_sections[section_header]
        return output  # type: ignore

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        output = defaultdict(list)
        for text in texts:
            hl_start, hl_end = None, None
            if text.citation_snippet_context and text.citation_snippet:
                hl_start, hl_end = DocumentCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=text.citation_snippet, context=text.citation_snippet_context
                )
            output[text].append(
                DocumentCitationOutput(
                    name=text.source_text.to_citation_title(),
                    summary=text.citation_snippet_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                    inline_offset=text.citation_text_offset,
                    last_updated_at=text.source_text.timestamp,
                )
            )

        return output  # type: ignore


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
    text_type: ClassVar[str] = "SEC Filing"
    db_id: Optional[str] = None
    form_type: Optional[str] = None

    @field_serializer("val")
    def serialize_val(self, val: str, _info: Any) -> str:
        # Make sure we don't serialize unnecessary data, we only want to serialize the ID.
        return ""

    @classmethod
    async def _get_strs_lookup(
        cls, sec_filing_list: List[StockSecFilingText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        """
        Complex logic to get the SEC filing text
        We're doing an on-demand caching for these SEC filings, meaning some of the filings can be
        found in the DB directly (db_id not None), and the rest will be fetched from the API.
        However, given the fact that we're separating the processes of getting the text ID and the
        text value, it's possible that some of the filings are already cached in the DB by other
        running agents before we actually download from API. With this context, do the following 3 steps:
        1. Get the filings that we know we can get from the DB first
        2. Using the remaining filing jsons (not db_id) to query DB and see if we can find cache
            This db query may not be optimal because we're using a large JSON as filters, but it should
            be better than hitting API anyway
        3. Download the rest from the API
        """

        filing_json_to_text_obj = {filing.id: filing for filing in sec_filing_list}

        output: Dict[TextIDType, str] = {}

        logger.info("Getting SEC filing text from DB using `db_id`")
        db_id_to_filing_json = {f.db_id: f.id for f in sec_filing_list if f.db_id}
        output.update(await SecFiling.get_concat_10k_10q_sections_from_db(db_id_to_filing_json))  # type: ignore
        logger.info(f"Found {len(output)} SEC filings in DB")

        logger.info("Getting SEC filing text from DB using filing json")
        # some db_id may be lost after clickhouse merges duplicates
        filing_jsons = [f.id for f in sec_filing_list if f.id not in output]
        filing_json_to_row = await SecFiling.get_concat_10k_10q_sections_from_db_by_filing_jsons(
            filing_jsons
        )
        for filing_json, (db_id, val) in filing_json_to_row.items():
            output[filing_json] = val

            if filing_json in filing_json_to_text_obj:
                text_obj = filing_json_to_text_obj[filing_json]
                text_obj.db_id = db_id  # set db_id

        logger.info(f"Found {len(filing_json_to_row)} SEC filings in DB using filing json")

        logger.info("Getting SEC filing text from API")
        filing_gbi_pairs = [
            (filing.id, filing.stock_id.gbi_id)
            for filing in sec_filing_list
            if not filing.db_id and filing.stock_id
        ]
        output.update(
            await SecFiling.get_concat_10k_10q_sections_from_api(filing_gbi_pairs, insert_to_db=True)  # type: ignore
        )

        return output

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        output = defaultdict(list)
        for text in texts:
            hl_start, hl_end = None, None
            if text.citation_snippet_context and text.citation_snippet:
                hl_start, hl_end = CompanyFilingCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=text.citation_snippet, context=text.citation_snippet_context
                )
            form_type = f" ({text.source_text.form_type})" if text.source_text.form_type else ""  # type: ignore
            if text.source_text.stock_id and text.source_text.timestamp:
                internal_id = CompanyFilingCitationOutput.get_citation_internal_id(
                    gbi_id=text.source_text.stock_id.gbi_id,
                    form_type=text.source_text.form_type,  # type: ignore
                    filing_date=text.source_text.timestamp.date(),
                )
            else:
                internal_id = str(text.source_text.db_id or text.source_text.id)  # type: ignore
            output[text].append(
                CompanyFilingCitationOutput(
                    internal_id=internal_id,
                    name=text.source_text.to_citation_title() + form_type,
                    summary=text.citation_snippet_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                    inline_offset=text.citation_text_offset,
                    last_updated_at=text.source_text.timestamp,
                )
            )
        return output  # type: ignore


@io_type
class StockSecFilingSectionText(StockText):
    """
    This class is actually a "section" of `StockSecFilingText`. Basically we will split the 2 sections
    (management, risk_factors) into even smaller sections and store them in this class.
    """

    id: Union[int, str]  # hash((self.filing_id, self.header))  (str for backwards compatibility)

    text_type: ClassVar[str] = "SEC Filing Section"

    filing_id: str
    header: str
    db_id: Optional[str] = None
    form_type: Optional[str] = None

    @field_serializer("val")
    def serialize_val(self, val: str, _info: Any) -> str:
        # Make sure we don't serialize unnecessary data, we only want to serialize the ID.
        return ""

    # For identifying the same texts across runs (different hash seeds)
    def reset_id(self) -> None:
        self.id = hash((self.filing_id, self.header))

    @classmethod
    async def init_from_full_text_data(
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
                        id=hash((filing.id, header)),
                        val=content,
                        filing_id=filing.id,
                        header=header,
                        stock_id=filing.stock_id,
                        db_id=filing.db_id,
                        timestamp=filing.timestamp,
                        form_type=filing.form_type,
                    )
                )

        return sections

    @classmethod
    async def _get_strs_lookup(
        cls, sections: List[StockSecFilingSectionText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        # Full data in the DB since we only store `header`, not `content` to save space
        filing_text_objs = {}
        header_to_sections: Dict[str, List[StockSecFilingSectionText]] = defaultdict(list)
        for section in sections:
            if section.filing_id not in filing_text_objs:
                filing_text_objs[section.filing_id] = StockSecFilingText(
                    id=section.filing_id,
                    stock_id=section.stock_id,
                    db_id=section.db_id,
                    timestamp=section.timestamp,
                    form_type=section.form_type,
                )

            # same header, multiple sections (across stocks)
            header_to_sections[section.header].append(section)

        filing_texts = await StockSecFilingText._get_strs_lookup(list(filing_text_objs.values()))

        outputs = {}
        for filing_id, filing_text in filing_texts.items():
            split_sections = SecFiling.split_10k_10q_into_smaller_sections(filing_text)
            for header, content in split_sections.items():
                text_str = f"{header}: {content}"
                for section in header_to_sections.get(header, []):
                    if section.filing_id == filing_id:
                        outputs[section.id] = text_str

        return outputs  # type: ignore

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        output = defaultdict(list)
        filing_id_set = set()
        for text in texts:
            source_text = text.source_text
            source_text = cast(Self, text.source_text)
            if source_text.filing_id in filing_id_set:
                continue
            filing_id_set.add(source_text.filing_id)
            hl_start, hl_end = None, None
            if text.citation_snippet_context and text.citation_snippet:
                hl_start, hl_end = CompanyFilingCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=text.citation_snippet, context=text.citation_snippet_context
                )
            form_type = f" ({text.source_text.form_type})" if text.source_text.form_type else ""  # type: ignore
            if text.source_text.stock_id and text.source_text.timestamp:
                internal_id = CompanyFilingCitationOutput.get_citation_internal_id(
                    gbi_id=text.source_text.stock_id.gbi_id,
                    form_type=text.source_text.form_type,  # type: ignore
                    filing_date=text.source_text.timestamp.date(),
                )
            else:
                internal_id = str(text.source_text.db_id or text.source_text.id)  # type: ignore
            output[text].append(
                CompanyFilingCitationOutput(
                    internal_id=internal_id,
                    name=text.source_text.to_citation_title() + form_type,
                    summary=text.citation_snippet_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                    inline_offset=text.citation_text_offset,
                    last_updated_at=text.source_text.timestamp,
                )
            )

        return output  # type: ignore

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return f"{self.header}: {self.val}"


@io_type
class StockOtherSecFilingText(StockSecFilingText):
    """
    Unlike `SecFilingText`, this class is used to get other types of SEC filings and the helper
    method `get_strs_lookup` is used to download the full content of the filing instead of only
    extracting a few sections.
    TODO: We may later merge this class with `SecFilingText` (and also extract certain sections
    for other types)
    """

    id: str  # SEC filing info
    text_type: ClassVar[str] = "SEC Filing"
    db_id: Optional[str] = None
    form_type: Optional[str] = None

    @field_serializer("val")
    def serialize_val(self, val: str, _info: Any) -> str:
        # Make sure we don't serialize unnecessary data, we only want to serialize the ID.
        return ""

    @classmethod
    async def _get_strs_lookup(
        cls, sec_filing_list: List[StockOtherSecFilingText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        """
        Complex logic to get the SEC filing text
        We're doing an on-demand caching for these SEC filings, meaning some of the filings can be
        found in the DB directly (db_id not None), and the rest will be fetched from the API.
        However, given the fact that we're separating the processes of getting the text ID and the
        text value, it's possible that some of the filings are already cached in the DB by other
        running agents before we actually download from API. With this context, do the following 3 steps:
        1. Get the filings that we know we can get from the DB first
        2. Using the remaining filing jsons (not db_id) to query DB and see if we can find cache
            This db query may not be optimal because we're using a large JSON as filters, but it should
            be better than hitting API anyway
        3. Download the rest from the API
        """

        filing_json_to_text_obj = {filing.id: filing for filing in sec_filing_list}

        # Get the available content from DB first
        output: Dict[TextIDType, str] = {}

        try:
            # determine which filings we know we definitely can get from the DB
            logger.info("Getting SEC filing text from DB using `db_id`")
            db_id_to_filing_json = {f.db_id: f.id for f in sec_filing_list if f.db_id}
            output.update(await SecFiling.get_filings_content_from_db(db_id_to_filing_json))  # type: ignore
            logger.info(f"Found {len(output)} SEC filings in DB")

            logger.info("Getting SEC filing text from DB using filing json")
            # some db_id may be lost after clickhouse merges duplicates
            filing_jsons = [f.id for f in sec_filing_list if f.id not in output]
            filing_json_to_row = await SecFiling.get_filings_content_from_db_by_filing_jsons(
                filing_jsons
            )
            for filing_json, (db_id, val) in filing_json_to_row.items():
                output[filing_json] = val

                if filing_json in filing_json_to_text_obj:
                    text_obj = filing_json_to_text_obj[filing_json]
                    text_obj.db_id = db_id  # set db_id

            logger.info(
                f"Found another {len(filing_json_to_row)} SEC filings in DB using filing json"
            )

            filing_gbi_pairs = [
                (filing.id, filing.stock_id.gbi_id)
                for filing in sec_filing_list
                if filing.id not in output and not filing.db_id and filing.stock_id
            ]
            if len(filing_gbi_pairs) > 0:
                logger.info(f"Getting the rest of {len(filing_gbi_pairs)} SEC filing text from API")
                output.update(
                    await SecFiling.get_filings_content_from_api(
                        filing_gbi_pairs, insert_to_db=True
                    )  # type: ignore
                )
                logger.info(f"Found another {len(filing_json_to_row)} SEC filings from API")
        except Exception as e:
            logger.info(f"Failed to retrieve text for SEC filings {e}")

        if len(output) != len(sec_filing_list):
            logger.info(
                f"Could not retrieve text for {len(sec_filing_list) - len(output)} SEC filings, defaulting empty"
            )
            output.update(
                {
                    filing.id: "Could not retrieve this filing."
                    for filing in sec_filing_list
                    if filing.id not in output
                }
            )

        return output

    async def to_rich_output(
        self,
        pg: BoostedPG,
        title: str = "",
        cached_resolved_citations: Optional[List[CitationOutput]] = None,
    ) -> Output:
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
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        output = defaultdict(list)
        for text in texts:
            hl_start, hl_end = None, None
            if text.citation_snippet_context and text.citation_snippet:
                hl_start, hl_end = CompanyFilingCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=text.citation_snippet, context=text.citation_snippet_context
                )
            form_type = f" ({text.source_text.form_type})" if text.source_text.form_type else ""  # type: ignore
            if text.source_text.stock_id and text.source_text.timestamp:
                internal_id = CompanyFilingCitationOutput.get_citation_internal_id(
                    gbi_id=text.source_text.stock_id.gbi_id,
                    form_type=text.source_text.form_type,  # type: ignore
                    filing_date=text.source_text.timestamp.date(),
                )
            else:
                internal_id = str(text.source_text.db_id or text.source_text.id)  # type: ignore
            output[text].append(
                CompanyFilingCitationOutput(
                    internal_id=internal_id,
                    name=text.source_text.to_citation_title() + form_type,
                    summary=text.citation_snippet_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                    inline_offset=text.citation_text_offset,
                    last_updated_at=text.source_text.timestamp,
                )
            )
        return output  # type: ignore


@io_type
class StockOtherSecFilingSectionText(StockText):
    """
    This class is actually a "section" of `StockOtherSecFilingText`. SEC filings are split
    into even smaller sections and store them in this class.
    """

    id: Union[int, str]  # hash((self.filing_id, self.header))  (str for backwards compatibility)

    text_type: ClassVar[str] = "SEC Filing Section"

    filing_id: str
    header: str
    db_id: Optional[str] = None
    form_type: Optional[str] = None

    @field_serializer("val")
    def serialize_val(self, val: str, _info: Any) -> str:
        # Make sure we don't serialize unnecessary data, we only want to serialize the ID.
        return ""

    # For identifying the same texts across runs (different hash seeds)
    def reset_id(self) -> None:
        self.id = hash((self.filing_id, self.header))

    @classmethod
    async def init_from_full_text_data(
        cls, filings: List[StockOtherSecFilingText]
    ) -> List[StockOtherSecFilingSectionText]:
        filing_texts = await StockOtherSecFilingText._get_strs_lookup(filings)

        sections = []
        for filing in filings:
            if filing.id not in filing_texts:
                continue

            split_sections = get_sections(filing_texts[filing.id])
            for header, content in split_sections.items():
                sections.append(
                    cls(
                        id=hash((filing.id, header)),
                        val=content,
                        filing_id=str(filing.id),
                        header=header,
                        stock_id=filing.stock_id,
                        db_id=filing.db_id,
                        timestamp=filing.timestamp,
                        form_type=filing.form_type,
                    )
                )

        return sections

    @classmethod
    async def _get_strs_lookup(
        cls, sections: List[StockOtherSecFilingSectionText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        # Full data in the DB since we only store `header`, not `content` to save space
        filing_text_objs = {}
        filing_to_section_ids: Dict[str, List[int]] = defaultdict(list)
        section_id_lookup = {section.id: section for section in sections}
        for section in sections:
            if section.filing_id not in filing_text_objs:
                filing_text_objs[section.filing_id] = StockOtherSecFilingText(
                    id=section.filing_id,
                    stock_id=section.stock_id,
                    db_id=section.db_id,
                    timestamp=section.timestamp,
                    form_type=section.form_type,
                )
            filing_to_section_ids[section.filing_id].append(int(section.id))

        filing_texts = await StockOtherSecFilingText._get_strs_lookup(
            list(filing_text_objs.values())
        )

        outputs = {}
        for filing_id, filing_text in filing_texts.items():
            split_sections = get_sections(filing_text)
            for section_id in filing_to_section_ids.get(str(filing_id), []):
                section = section_id_lookup[section_id]
                outputs[section.id] = split_sections.get(section.header)

        return outputs  # type: ignore

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        output = defaultdict(list)
        filing_id_set = set()
        for text in texts:
            source_text = text.source_text
            source_text = cast(Self, text.source_text)
            if source_text.filing_id in filing_id_set:
                continue
            filing_id_set.add(source_text.filing_id)
            hl_start, hl_end = None, None
            if text.citation_snippet_context and text.citation_snippet:
                hl_start, hl_end = CompanyFilingCitationOutput.get_offsets_from_snippets(
                    smaller_snippet=text.citation_snippet, context=text.citation_snippet_context
                )
            form_type = f" ({text.source_text.form_type})" if text.source_text.form_type else ""  # type: ignore
            if text.source_text.stock_id and text.source_text.timestamp:
                internal_id = CompanyFilingCitationOutput.get_citation_internal_id(
                    gbi_id=text.source_text.stock_id.gbi_id,
                    form_type=text.source_text.form_type,  # type: ignore
                    filing_date=text.source_text.timestamp.date(),
                )
            else:
                internal_id = str(text.source_text.db_id or text.source_text.id)  # type: ignore
            output[text].append(
                CompanyFilingCitationOutput(
                    internal_id=internal_id,
                    name=text.source_text.to_citation_title() + form_type,
                    summary=text.citation_snippet_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                    inline_offset=text.citation_text_offset,
                    last_updated_at=text.source_text.timestamp,
                )
            )

        return output  # type: ignore

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return f"{self.header}: {self.val}"


@io_type
class WebText(Text):
    text_type: ClassVar[str] = "Web Results"
    url: Optional[str] = None
    title: Optional[str] = ""

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        output = defaultdict(list)
        for cit in texts:
            text = cast(Self, cit.source_text)

            full_context = cit.citation_snippet_context
            snippet = cit.citation_snippet
            hl_start, hl_end = DocumentCitationOutput.get_offsets_from_snippets(
                smaller_snippet=snippet, context=full_context
            )

            output[cit].append(
                WebCitationOutput(
                    internal_id=str(text.id),
                    name=text.title or text.url or text.to_citation_title(),
                    link=text.url,
                    inline_offset=cit.citation_text_offset,
                    summary=full_context,
                    snippet_highlight_start=hl_start,
                    snippet_highlight_end=hl_end,
                )
            )
        return output  # type: ignore


@io_type
class KPIText(Text):
    pid: Optional[int] = None
    explanation: Optional[str] = None  # To be used as a way to store why a KPI has been selected
    text_type: ClassVar[str] = "KPI"
    url: Optional[str] = None

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[TextCitation], db: BoostedPG
    ) -> Dict[TextCitation, Sequence[CitationOutput]]:
        output = defaultdict(list)
        for cit in texts:
            text = cast(Self, cit.source_text)
            output[cit].append(
                KPICitationOutput(
                    internal_id=str(text.id), name=text.val, summary=text.explanation, link=text.url
                )
            )
        return output  # type: ignore


# These are not actual Text types, but build on top of them


@io_type
class TextGroup(ComplexIOBase):
    val: List[Text]
    id_to_str: Optional[Dict[TextIDType, str]] = None
    offset: int = 0  # for starting the numbering of a TextGroup from something other than 0

    @staticmethod
    def join(group1: TextGroup, group2: TextGroup) -> TextGroup:
        texts = group1.val + group2.val
        joined_id_to_str = {
            **(group1.id_to_str if group1.id_to_str else {}),
            **(group2.id_to_str if group2.id_to_str else {}),
        }
        return TextGroup(val=texts, id_to_str=joined_id_to_str)

    def convert_to_str(
        self,
        id_to_str: Dict[TextIDType, str],
        numbering: bool = False,
    ) -> str:
        # sort texts by timestamp so any truncations cuts off older texts
        self.val.sort(key=lambda x: x.timestamp.timestamp() if x.timestamp else 0, reverse=True)
        self.id_to_str = id_to_str
        return "\n***\n".join(
            [
                (f"Text Number: {i + self.offset}\n" if numbering else "") + id_to_str[text.id]
                for i, text in enumerate(self.val)
                if text.id in id_to_str
            ]
        )

    def convert_citation_num_to_text(self, citation_id: int) -> Optional[Text]:
        try:
            return self.val[int(citation_id) - self.offset]
        except (ValueError, IndexError):
            logger.exception("Could not convert citation num to text")
            return None

    def get_str_for_text(self, text_id: TextIDType) -> Optional[str]:
        try:
            return self.id_to_str[text_id]  # type: ignore
        except (TypeError, KeyError):
            # can't log this due to circular import
            logger.exception("Could not convert text ID to text")
            return None

    def get_citations(self, citation_ids: List[int]) -> List[Citation]:
        # do int(i) just in case GPT screwed up
        return [
            TextCitation(source_text=self.val[int(i) - self.offset])
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
    initial_idea: Optional[str] = None  # Stores the initial idea that spawned these profiles


class TextOutput(Output):
    output_type: Literal[OutputType.TEXT] = OutputType.TEXT
    val: str
    # Optional top level score for this segment of text
    score: Optional[ScoreOutput] = None
    resolved_text_objects: bool = False

    def render_text_objects(self, text_objects: Optional[List[TextObject]] = None) -> None:
        """
        Given a string representing this text's value, go through all the
        citations and text objects for this text and insert them into the value
        string.
        """
        if not text_objects:
            return

        if self.resolved_text_objects or """```{{ "type": """ in self.val:
            # There are times that this function may be called more than once on
            # the same object. This will prevent any issues or duplicate
            # citations.
            return

        self.resolved_text_objects = True

        self.val = CitationTextObject.render_text_objects(text=self.val, objects=text_objects)

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
    async def resolve_citations(
        cls, citations: List[Self], db: BoostedPG
    ) -> Dict[Self, List[CitationOutput]]:
        # First group citations based on type of the source text.
        text_type_map = defaultdict(list)
        for cit in citations:
            text_type_map[type(cit.source_text)].append(cit)

        tasks = [
            typ.get_citations_for_output(text_list, db) for typ, text_list in text_type_map.items()  # type: ignore
        ]
        outputs = defaultdict(list)
        # List of lists, where each nested list has outputs for each type
        outputs_nested: List[Dict[TextCitation, List[CitationOutput]]] = (
            await gather_with_concurrency(tasks)
        )
        # Unnest into a single list
        for output_dict in outputs_nested:
            for citation, output_cits in output_dict.items():
                outputs[citation].extend(output_cits)

        return outputs  # type: ignore


@io_type
class TextCitationGroup(ComplexIOBase):
    val: List[TextCitation]

    def _sort_citations(self) -> None:
        self.val = sorted(
            self.val,
            key=lambda x: (
                1 if x.citation_snippet is None else 0,
                x.source_text.timestamp.timestamp() if x.source_text.timestamp else 0,
            ),
            reverse=True,
        )

    def _get_snippet_start(self) -> int:
        try:
            return self.val.index(next(x for x in self.val if x.citation_snippet is not None))
        except StopIteration:
            return len(self.val)

    async def convert_to_str(self) -> str:
        self._sort_citations()
        snippet_start = self._get_snippet_start()
        no_snippet_str = cast(
            str,
            await Text.get_all_strs(
                TextGroup(val=[citation.source_text for citation in self.val[:snippet_start]]),
                text_group_numbering=True,
            ),
        )
        full_str = "\n***\n".join(
            [no_snippet_str]
            + [
                f"Text Number: {i + snippet_start}\n{citation.citation_snippet}"
                for i, citation in enumerate(self.val[snippet_start:])
            ]
        )
        return full_str

    def convert_citation_num_to_citation(self, citation_id: int) -> Optional[TextCitation]:
        try:
            return self.val[int(citation_id)]
        except (ValueError, IndexError):
            logger.exception("Could not convert citation num to citation")
            return None
