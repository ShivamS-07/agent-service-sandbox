import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HypothesisInfo:
    hypothesis_text: str

    gbi_id: int
    company_name: str
    company_description: str

    hypothesis_breakdown: Optional[Dict[str, Any]] = None  # preprocessed result from text

    embedding: Optional[List[float]] = None  # embedding of text + breakdown

    hypothesis_summary: Optional[str] = None

    match_scores: Optional[List[Tuple[datetime.date, float]]] = None


class Polarity(int, Enum):
    positive = 1
    neutral = 0
    negative = -1


class NewsImpact(int, Enum):
    low = 0
    medium = 1
    high = 2


@dataclass
class CompanyTopicInfo:
    topic_id: str
    topic_label: str
    topic_descriptions: List[Tuple[str, datetime.datetime]]
    topic_polarities: List[Tuple[Polarity, datetime.datetime]]
    gbi_id: int

    def get_latest_topic_polarity(self, default: Optional[Polarity] = None) -> Optional[Polarity]:
        return self.topic_polarities[-1][0] if self.topic_polarities else default

    def get_latest_topic_description(self, default: Optional[str] = None) -> Optional[str]:
        return self.topic_descriptions[-1][0] if self.topic_descriptions else default


@dataclass
class CompanyNewsTopicInfo(CompanyTopicInfo):
    gbi_id: int = -1  # -1 is used for topics with no set company
    topic_query: Optional[str] = None  # query for getting topic relevant news, needs to inserted
    is_indirect: bool = (
        False  # topic is from cross company pool, field does not need to be inserted into db
    )
    rationale: Optional[str] = (
        None  # contains the rationale for cross company topics, not for db insert
    )
    topic_impacts: List[Tuple[NewsImpact, datetime.datetime]] = field(default_factory=list)

    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

    original_topic_polarities: Optional[List[Tuple[Polarity, datetime.datetime]]] = None
    original_topic_impacts: Optional[List[Tuple[NewsImpact, datetime.datetime]]] = None

    def get_latest_original_topic_impact(
        self, default: Optional[NewsImpact] = None
    ) -> Optional[NewsImpact]:
        if self.original_topic_impacts:
            return self.original_topic_impacts[-1][0]
        return default

    def get_latest_original_topic_polarity(
        self, default: Optional[Polarity] = None
    ) -> Optional[Polarity]:
        return self.original_topic_polarities[-1][0] if self.original_topic_polarities else default

    def get_latest_topic_impact(self, default: Optional[NewsImpact] = None) -> Optional[NewsImpact]:
        return self.topic_impacts[-1][0] if self.topic_impacts else default


class EarningsSummaryType(str, Enum):
    REMARKS = "Remarks"
    QUESTIONS = "Questions"
    HIGHLIGHTS = "highlights"
    PEERS = "Peers"


@dataclass
class CompanyEarningsTopicInfo(CompanyTopicInfo):
    summary_index: int
    summary_type: EarningsSummaryType
    summary_date: datetime.datetime
    topic_impacts: List[Tuple[NewsImpact, datetime.datetime]] = field(default_factory=list)

    peer_company_gbi_id: Optional[int] = None

    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None
    year: Optional[int] = None
    quarter: Optional[int] = None

    def get_latest_topic_impact(self, default: Optional[NewsImpact] = None) -> Optional[NewsImpact]:
        return self.topic_impacts[-1][0] if self.topic_impacts else default


@dataclass
class HypothesisNewsTopicInfo:
    """
    To convert to Text object, we need such fields:

    """

    gbi_id: int
    topic_id: str

    hypothesis_topic_supports: List[
        Tuple[float, datetime.datetime]
    ]  # [-1, 1], negative means contradict

    hypothesis_topic_impacts: List[Tuple[NewsImpact, datetime.datetime]] = field(
        default_factory=list
    )
    hypothesis_topic_polarities: List[Tuple[Polarity, datetime.datetime]] = field(
        default_factory=list
    )
    hypothesis_topic_reasons: List[Tuple[str, datetime.datetime]] = field(default_factory=list)

    def get_latest_support(self, default: Optional[float] = None) -> Optional[float]:
        return self.hypothesis_topic_supports[-1][0] if self.hypothesis_topic_supports else default

    def get_latest_impact(self, default: Optional[NewsImpact] = None) -> Optional[NewsImpact]:
        return self.hypothesis_topic_impacts[-1][0] if self.hypothesis_topic_impacts else default

    def get_latest_polarity(self, default: Optional[Polarity] = None) -> Optional[Polarity]:
        return (
            self.hypothesis_topic_polarities[-1][0] if self.hypothesis_topic_polarities else default
        )

    def get_latest_reason(self, default: Optional[str] = None) -> Optional[str]:
        return self.hypothesis_topic_reasons[-1][0] if self.hypothesis_topic_reasons else default


@dataclass
class HypothesisEarningsTopicInfo:
    gbi_id: int

    topic_id: str
    summary_index: int
    summary_type: EarningsSummaryType
    summary_date: datetime.datetime

    hypothesis_topic_supports: List[
        Tuple[float, datetime.datetime]
    ]  # [-1, 1], negative means contradict

    hypothesis_topic_impacts: List[Tuple[int, datetime.datetime]] = field(default_factory=list)
    hypothesis_topic_polarities: List[Tuple[Polarity, datetime.datetime]] = field(
        default_factory=list
    )
    hypothesis_topic_reasons: List[Tuple[str, datetime.datetime]] = field(default_factory=list)

    peer_gbi_id: Optional[int] = None

    def get_latest_support(self, default: Optional[float] = None) -> Optional[float]:
        return self.hypothesis_topic_supports[-1][0] if self.hypothesis_topic_supports else default

    def get_latest_impact(self, default: Optional[int] = None) -> Optional[int]:
        return self.hypothesis_topic_impacts[-1][0] if self.hypothesis_topic_impacts else default

    def get_latest_reason(self, default: Optional[str] = None) -> Optional[str]:
        return self.hypothesis_topic_reasons[-1][0] if self.hypothesis_topic_reasons else default

    def get_latest_polarity(self, default: Optional[Polarity] = None) -> Optional[Polarity]:
        return (
            self.hypothesis_topic_polarities[-1][0] if self.hypothesis_topic_polarities else default
        )


@dataclass
class CompanyNewsInfo:
    news_id: str

    headline: str
    # summary: str
    # url: str

    published_at: datetime.datetime
    gbi_id: int

    topic_id: Optional[str] = None
    is_top_source: bool = False  # True if this is a top 25 finance source
