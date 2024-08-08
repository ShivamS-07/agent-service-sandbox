import datetime
import enum
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.sec.sec_api import SecFiling


class CitationType(str, enum.Enum):
    CUSTOM_DOC = "custom_doc"
    THEME = "theme"
    NEWS_DEVELOPMENT = "news_development"
    NEWS_ARTICLE = "news_article"
    COMPANY_FILING = "company_filing"
    EARNINGS_SUMMARY = "earnings_summary"

    # Generic, should no longer be used
    LINK = "link"
    TEXT = "text"

    def to_citation_class(self) -> Type["CitationOutput"]:
        if self == CitationType.CUSTOM_DOC:
            return CustomDocumentCitationOutput
        elif self == CitationType.THEME:
            return ThemeCitationOutput
        elif self == CitationType.NEWS_DEVELOPMENT:
            return NewsDevelopmentCitationOutput
        elif self == CitationType.NEWS_ARTICLE:
            return NewsArticleCitationOutput
        elif self == CitationType.COMPANY_FILING:
            return CompanyFilingCitationOutput
        elif self == CitationType.EARNINGS_SUMMARY:
            return EarningsSummaryCitationOutput

        return CitationOutput


CitationID = str


class CitationDetails(BaseModel):
    citation_id: CitationID
    citation_type: CitationType
    title: str


class RawTextCitationDetails(CitationDetails):
    raw_text: str


class NewsDevelopmentCitationDetails(CitationDetails):
    class ArticleInfo(BaseModel):
        headline: str
        published_at: datetime.datetime
        url: str
        source: Optional[str] = None

    num_articles: int
    last_updated: datetime.datetime
    summary: Optional[str] = None
    articles: List[ArticleInfo] = Field(default_factory=list)


CitationDetailsType = Union[RawTextCitationDetails, NewsDevelopmentCitationDetails, CitationDetails]


class CitationOutput(BaseModel, ABC):
    id: CitationID = Field(default_factory=lambda: str(uuid4()))
    internal_id: CitationID = Field(default_factory=lambda: str(uuid4()))
    citation_type: CitationType
    name: str
    inline_offset: Optional[int] = None
    summary: Optional[str] = None
    last_updated_at: Optional[datetime.datetime] = None
    is_snippet: bool = False

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)

    @classmethod
    async def get_citation_details(
        cls, citation_id: str, db: BoostedPG
    ) -> Optional[CitationDetails]:
        """
        Given a citation ID, converts the ID to a relevant CitationDetails
        object that can be shown to the frontend.
        """
        return None


class DocumentCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.TEXT
    snippet_highlight_start: Optional[int] = None
    snippet_highlight_end: Optional[int] = None  # inclusive
    is_snippet: bool = True

    @staticmethod
    def get_offsets_from_snippets(
        smaller_snippet: Optional[str], context: Optional[str]
    ) -> Tuple[Optional[int], Optional[int]]:
        if not smaller_snippet or not context:
            return (None, None)

        try:
            start = context.index(smaller_snippet)
            end = start + len(smaller_snippet) - 1
            return (start, end)
        except ValueError:
            return (None, None)


class ThemeCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.THEME


class CustomDocumentCitationOutput(DocumentCitationOutput):
    citation_type: CitationType = CitationType.CUSTOM_DOC
    custom_doc_id: str


class CompanyFilingCitationOutput(DocumentCitationOutput):
    citation_type: CitationType = CitationType.COMPANY_FILING

    @classmethod
    async def get_citation_details(
        cls, citation_id: str, db: BoostedPG
    ) -> Optional[RawTextCitationDetails]:
        from agent_service.io_types.stock import StockID

        result = await SecFiling.get_filing_data_async(db_ids=[citation_id])
        if not result:
            return None
        sec_data = result[citation_id]
        stock = (await StockID.from_gbi_id_list([sec_data.gbi_id]))[0]
        filed_at_str = sec_data.filed_at.strftime("%Y-%m-%d")
        title = f"{stock.company_name} - {sec_data.form_type} ({filed_at_str})"
        return RawTextCitationDetails(
            citation_id=citation_id,
            citation_type=CitationType.COMPANY_FILING,
            title=title,
            raw_text=sec_data.content,
        )


class NewsDevelopmentCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.NEWS_DEVELOPMENT
    num_articles: Optional[int] = None

    @classmethod
    async def get_citation_details(
        cls, citation_id: str, db: BoostedPG
    ) -> Optional[NewsDevelopmentCitationDetails]:
        sql = """
        (
        SELECT topic_id::TEXT AS citation_id, TRUE AS is_stock_topic,
               topic_label AS title, (topic_descriptions->-1->>0)::TEXT AS summary
        FROM nlp_service.stock_news_topics
        WHERE topic_id = %(citation_id)s
        )
        UNION
        (
        SELECT development_id::TEXT AS citation_id, FALSE AS is_stock_topic,
               label AS title, description AS summary
        FROM nlp_service.theme_developments
        WHERE development_id = %(citation_id)s
        )
        """
        rows = await db.generic_read(sql, {"citation_id": citation_id})
        if not rows:
            return None
        if rows[0]["is_stock_topic"]:
            sql = """
            SELECT sn.headline, sn.published_at, sn.url, s.domain_url AS source
            FROM nlp_service.stock_news sn
            JOIN nlp_service.news_sources s ON sn.source_id=s.source_id
            WHERE topic_id = %(citation_id)s
            ORDER BY s.is_top_source DESC, sn.published_at DESC
            """
        else:
            sql = """
            SELECT tn.headline, tn.published_at, tn.url, s.domain_url AS source
            FROM nlp_service.theme_news tn
            JOIN nlp_service.news_sources s ON tn.source_id=s.source_id
            WHERE development_id = %(citation_id)s
            ORDER BY s.is_top_source DESC, tn.published_at DESC
            """

        news_rows = await db.generic_read(sql, {"citation_id": citation_id})
        if not news_rows:
            return None
        last_updated = max((row["published_at"] for row in news_rows))

        return NewsDevelopmentCitationDetails(
            title=rows[0]["title"],
            summary=rows[0]["summary"],
            citation_id=citation_id,
            citation_type=CitationType.NEWS_DEVELOPMENT,
            num_articles=len(news_rows),
            last_updated=last_updated,
            articles=[NewsDevelopmentCitationDetails.ArticleInfo(**row) for row in news_rows],
        )


class NewsArticleCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.NEWS_ARTICLE
    link: Optional[str] = None
    article_id: Optional[str] = None


class EarningsSummaryCitationOutput(DocumentCitationOutput):
    citation_type: CitationType = CitationType.EARNINGS_SUMMARY


# Deprecated
class TextCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.TEXT


class LinkCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.LINK
    link: Optional[str] = None
