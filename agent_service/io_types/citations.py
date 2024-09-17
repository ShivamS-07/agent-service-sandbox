import datetime
import enum
import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import uuid4

from grpclib import GRPCError
from pydantic import BaseModel, Field

from agent_service.external.custom_data_svc_client import (
    get_citation_custom_doc_context,
)
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.clickhouse import AsyncClickhouseBase
from agent_service.utils.sec.sec_api import SecFiling

logger = logging.getLogger(__name__)


class CitationType(enum.StrEnum):
    CUSTOM_DOC = "custom_doc"
    THEME = "theme"
    NEWS_DEVELOPMENT = "news_development"
    NEWS_ARTICLE = "news_article"
    COMPANY_FILING = "company_filing"
    EARNINGS_SUMMARY = "earnings_summary"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    KPI = "kpi"

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
        elif self == CitationType.EARNINGS_TRANSCRIPT:
            return EarningsTranscriptCitationOutput
        elif self == CitationType.KPI:
            return KPICitationOutput

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


class CustomDocumentCitationDetails(CitationDetails):
    class CustomDocumentTxtCitationLocation(BaseModel):
        class TxtCitationHighlight(BaseModel):
            start_line_idx: int
            start_char: int
            end_char: int

        highlights: List[TxtCitationHighlight]

    class CustomDocumentPdfCitationLocation(BaseModel):
        class PdfCitationHighlight(BaseModel):
            page_number: int
            bbox_x0: float
            bbox_y0: float
            bbox_x1: float
            bbox_y1: float

        highlights: List[PdfCitationHighlight]

    long_summary_offset: int
    citation_snippet: str
    citation_context: str
    citation_source_loc: CustomDocumentPdfCitationLocation | CustomDocumentTxtCitationLocation
    custom_doc_id: str
    custom_doc_file_type: str
    chunk_id: str
    long_summary: str


CitationDetailsType = Union[
    CustomDocumentCitationDetails,
    RawTextCitationDetails,
    NewsDevelopmentCitationDetails,
    CitationDetails,
]


####################################################################################################
# Citation Details
####################################################################################################
class GetCitationDetailsResponse(BaseModel):
    details: Optional[CitationDetailsType]


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
        cls, citation_id: str, db: BoostedPG, user_id: str
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
    chunk_id: str

    @classmethod
    async def get_citation_details(
        cls, citation_id: str, db: BoostedPG, user_id: str
    ) -> Optional[CustomDocumentCitationDetails]:
        try:
            detail_resp = await get_citation_custom_doc_context(
                citation_id=citation_id, user_id=user_id
            )
            citation_info = detail_resp.citation_info

            return CustomDocumentCitationDetails(
                citation_id=citation_id,
                citation_type=CitationType.CUSTOM_DOC,
                custom_doc_id=citation_info.file_id,
                custom_doc_file_type=citation_info.document_file_type,
                chunk_id=citation_info.article_id,
                title=f"User document: {citation_info.file_name.lstrip("/")}",
                long_summary=citation_info.long_summary,
                long_summary_offset=citation_info.citation.long_summary_offset,
                citation_context=citation_info.citation.citation_context,
                citation_snippet=citation_info.citation.citation_snippet,
                citation_source_loc=(
                    CustomDocumentCitationDetails.CustomDocumentPdfCitationLocation(
                        highlights=[
                            CustomDocumentCitationDetails.CustomDocumentPdfCitationLocation.PdfCitationHighlight(
                                page_number=h.page_number,
                                bbox_x0=h.bbox_x0,
                                bbox_x1=h.bbox_x1,
                                bbox_y0=h.bbox_y0,
                                bbox_y1=h.bbox_y1,
                            )
                            for h in citation_info.citation.pdf_citation_loc.highlights
                        ]
                    )
                    if citation_info.citation.HasField("pdf_citation_loc")
                    else CustomDocumentCitationDetails.CustomDocumentTxtCitationLocation(
                        highlights=[
                            CustomDocumentCitationDetails.CustomDocumentTxtCitationLocation.TxtCitationHighlight(
                                start_line_idx=h.line_idx,
                                start_char=h.start_char,
                                end_char=h.end_char,
                            )
                            for h in citation_info.citation.txt_citation_loc.highlights
                        ]
                    )
                ),
            )

        except GRPCError as e:
            logger.exception(f"Error getting citation {citation_id} for custom doc: {e}")
            return None


class CompanyFilingCitationOutput(DocumentCitationOutput):
    citation_type: CitationType = CitationType.COMPANY_FILING

    @classmethod
    def get_citation_internal_id(
        cls, gbi_id: int, form_type: str, filing_date: datetime.date
    ) -> str:
        date_str = filing_date.strftime("%Y_%m_%d")
        # For backwards compatibility, if the form type doesn't exist
        return f"{gbi_id}_{form_type}_{date_str}"

    @classmethod
    def parse_id(cls, citation_id: str) -> Union[Tuple[int, str, datetime.date], str]:
        """
        Earnings transcripts may be identifier EITHER by their gbi id, year,
        quarter OR by the transcript ID for the record in clickhouse. This
        function parses the identifier and returns one of those options.
        """
        parts = citation_id.split("_")
        if len(parts) == 5:
            # gbi id, form type, year, month, date
            gbi_id = int(parts[0])
            form_type = parts[1]
            date = datetime.date(int(parts[2]), int(parts[3]), int(parts[4]))
            return (gbi_id, form_type, date)
        else:
            # It's just a UUID, use as is
            return citation_id

    @classmethod
    async def get_citation_details(
        cls, citation_id: str, db: BoostedPG, user_id: str
    ) -> Optional[RawTextCitationDetails]:
        parsed_id = CompanyFilingCitationOutput.parse_id(citation_id)

        from agent_service.io_types.stock import StockID

        if isinstance(parsed_id, tuple):
            sec_data = await SecFiling.get_filing_data_by_type_date_async(
                gbi_id=parsed_id[0], filing_type=parsed_id[1], date=parsed_id[2]
            )
            if not sec_data:
                return None
        else:
            # It's a UUID
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
        cls, citation_id: str, db: BoostedPG, user_id: str
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


class EarningsTranscriptCitationOutput(DocumentCitationOutput):
    citation_type: CitationType = CitationType.EARNINGS_TRANSCRIPT

    @classmethod
    def get_citation_internal_id(cls, gbi_id: int, year: int, quarter: int) -> str:
        return f"{gbi_id}_{year}_{quarter}"

    @classmethod
    def parse_id(cls, citation_id: str) -> Union[Tuple[int, int, int], str]:
        """
        Earnings transcripts may be identifier EITHER by their gbi id, year,
        quarter OR by the transcript ID for the record in clickhouse. This
        function parses the identifier and returns one of those options.
        """
        parts = citation_id.split("_")
        if len(parts) == 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        else:
            # It's just a UUID, use as is
            return citation_id

    @classmethod
    async def get_citation_details(
        cls, citation_id: str, db: BoostedPG, user_id: str
    ) -> Optional[RawTextCitationDetails]:
        from agent_service.io_types.stock import StockID

        parsed_id = cls.parse_id(citation_id)

        ch = AsyncClickhouseBase()
        if isinstance(parsed_id, tuple):
            earnings_transcript_sql = """
                SELECT gbi_id, earnings_date, transcript
                FROM company_earnings.full_earning_transcripts
                WHERE gbi_id = %(gbi_id)s AND fiscal_year=%(year)s AND fiscal_quarter=%(quarter)s
                ORDER BY inserted_time DESC
                LIMIT 1
            """
            rows = await ch.generic_read(
                earnings_transcript_sql,
                {"gbi_id": parsed_id[0], "year": parsed_id[1], "quarter": parsed_id[2]},
            )
        else:
            earnings_transcript_sql = """
                SELECT gbi_id, earnings_date, transcript
                FROM company_earnings.full_earning_transcripts
                WHERE id = %(citation_id)s
            """
            rows = await ch.generic_read(
                earnings_transcript_sql,
                {"citation_id": citation_id},
            )
        if not rows:
            return None

        row = rows[0]
        gbi_id = row["gbi_id"]
        stock = (await StockID.from_gbi_id_list([gbi_id]))[0]
        earnings_date_str = row["earnings_date"].strftime("%Y-%m-%d")
        title = f"{stock.company_name} - Earnings Call Transcript ({earnings_date_str})"

        return RawTextCitationDetails(
            citation_id=citation_id,
            citation_type=CitationType.EARNINGS_TRANSCRIPT,
            title=title,
            raw_text=row["transcript"],
        )


class KPICitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.KPI
    link: Optional[str] = None


# Deprecated
class TextCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.TEXT


class LinkCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.LINK
    link: Optional[str] = None
