import datetime
import enum
from abc import ABC
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


class CitationType(str, enum.Enum):
    CUSTOM_DOC = "custom_doc"
    THEME = "theme"
    NEWS_DEVELOPMENT = "news_development"
    NEWS_ARTICLE = "news_article"
    COMPANY_FILING = "company_filing"

    # Generic, should no longer be used
    LINK = "link"
    TEXT = "text"


CitationID = str


class CitationOutput(BaseModel, ABC):
    id: CitationID = Field(default_factory=lambda: str(uuid4()))
    citation_type: CitationType
    name: str
    inline_offset: Optional[int] = None

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)


class DocumentCitationOutput(CitationOutput, ABC):
    cited_snippet: Optional[str] = None
    snippet_highlight_start: Optional[int] = None
    snippet_highlight_end: Optional[int] = None

    @staticmethod
    def get_offsets_from_snippets(
        smaller_snippet: str, context: str
    ) -> Tuple[Optional[int], Optional[int]]:
        try:
            start = context.index(smaller_snippet)
            end = len(smaller_snippet) - 1
            return (start, end)
        except ValueError:
            return (None, None)


class ThemeCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.THEME
    summary: Optional[str] = None
    last_updated_at: Optional[datetime.datetime] = None


class CustomDocumentCitationOutput(DocumentCitationOutput):
    citation_type: CitationType = CitationType.CUSTOM_DOC
    custom_doc_id: str
    last_updated_at: Optional[datetime.datetime] = None


class CompanyFilingCitationOutput(DocumentCitationOutput):
    citation_type: CitationType = CitationType.COMPANY_FILING


class NewsDevelopmentCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.NEWS_DEVELOPMENT
    summary: Optional[str] = None
    last_updated_at: Optional[datetime.datetime] = None
    num_articles: Optional[int] = None


class NewsArticleCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.NEWS_ARTICLE
    link: Optional[str] = None
    summary: Optional[str] = None
    last_updated_at: Optional[datetime.datetime] = None
    article_id: Optional[str] = None


# For backwards compatibility only, should not be used for new things
class TextCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.TEXT
    published_at: Optional[datetime.datetime] = None
    summary: Optional[str] = None


class LinkCitationOutput(CitationOutput):
    citation_type: CitationType = CitationType.LINK
    published_at: Optional[datetime.datetime] = None
    summary: Optional[str] = None
    link: Optional[str] = None
