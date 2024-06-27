from __future__ import annotations

import json
from collections import defaultdict
from itertools import chain
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union
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
from agent_service.io_types.output import (
    CitationOutput,
    CitationType,
    Output,
    OutputType,
)
from agent_service.io_types.stock import StockID
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.sec.constants import (
    LINK_TO_HTML,
    MANAGEMENT_SECTION,
    RISK_FACTORS,
)
from agent_service.utils.sec.sec_api import SecFiling

TextIDType = Union[str, int]


@io_type
class Text(ComplexIOBase):
    id: TextIDType = Field(default_factory=lambda: str(uuid4()))
    val: str = ""
    text_type: ClassVar[str] = "Misc"
    stock_id: Optional[StockID] = None

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
            self.get_citations_for_output(texts=[self], db=pg),
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
        text = await self.get()
        return f"<Text: {text.val}>"

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
        strs_lookup: Dict[TextIDType, str] = {}
        # For every subclass of Text, fetch data
        lookups: List[Dict[TextIDType, str]] = await gather_with_concurrency(
            [
                textclass.get_strs_lookup(texts, include_header=include_header)
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
        cls, texts: List[Self], include_header: bool = False
    ) -> Dict[TextIDType, str]:
        strs_lookup = await cls._get_strs_lookup(texts)
        if include_header:
            for id, val in strs_lookup.items():
                strs_lookup[id] = f"Text type: {cls.text_type}\nText:\n{val}"
        return strs_lookup

    @classmethod
    async def _get_strs_lookup(cls, texts: List[Self]) -> Dict[TextIDType, str]:
        return {text.id: text.val for text in texts}

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        """
        Given a list of texts of this type, return a list of citations related
        the the input texts.
        By default, return no citations for the text.
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
        return {row["topic_id"]: f"{row['topic_label']}:\n{row['description']}" for row in rows}

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        sql = """
        SELECT DISTINCT ON (sn.topic_id)
          news_id::TEXT, url, domain_url, headline, published_at
        FROM nlp_service.stock_news sn
        JOIN nlp_service.news_sources ns ON ns.source_id = sn.source_id
        WHERE sn.topic_id = ANY(%(topic_ids)s)
        ORDER BY sn.topic_id, sn.is_top_source DESC, published_at DESC
        """
        params = {"topic_ids": [text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return [
            CitationOutput(
                id=row["news_id"],
                citation_type=CitationType.LINK,
                name=row["domain_url"],
                link=row["url"],
                summary=row["headline"],
                published_at=row["published_at"],
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
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        sql = """
        SELECT news_id::TEXT, url, domain_url, headline, published_at
        FROM nlp_service.stock_news sn
        JOIN nlp_service.news_sources ns ON ns.source_id = sn.source_id
        WHERE sn.news_id = ANY(%(news_ids)s)
        """
        params = {"news_ids": [text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return [
            CitationOutput(
                id=row["news_id"],
                citation_type=CitationType.LINK,
                name=row["domain_url"],
                link=row["url"],
                summary=row["headline"],
                published_at=row["published_at"],
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
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        sql = """
        SELECT news_id::TEXT, url, domain_url, headline, published_at
        FROM nlp_service.news_pool np
        JOIN nlp_service.news_sources ns ON ns.source_id = np.source_id
        WHERE np.news_id = ANY(%(news_ids)s)
        """
        params = {"news_ids": [text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return [
            CitationOutput(
                id=row["news_id"],
                citation_type=CitationType.LINK,
                name=row["domain_url"],
                link=row["url"],
                summary=row["headline"],
                published_at=row["published_at"],
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
                texts[id] = f"{chunk_info.headline}\n{chunk_info.summary}"
        return texts

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        # group by user id - we are assuming that anyone who is authed for the agent
        # has priv to see any documents utilized by the agent.
        articles_text_by_user: Dict[str, List[CustomDocumentSummaryText]] = defaultdict(list)
        for doc in texts:
            articles_text_by_user[doc.requesting_user].append(doc)

        citations = []
        for user, articles in articles_text_by_user.items():
            article_info = await get_custom_doc_articles_info(
                user, [article.id for article in articles]
            )
            for id, chunk_info in dict(article_info.file_chunk_info).items():
                file_paths = list(chunk_info.file_paths)
                if len(file_paths) == 0:
                    citation_name = chunk_info.file_id
                else:
                    # pick an arbitrary custom file path
                    citation_name = file_paths[0]

                citations.append(
                    CitationOutput(
                        id=id,
                        citation_type=CitationType.TEXT,
                        name=f"User Document: {citation_name}",
                        published_at=chunk_info.upload_time.ToDatetime(),
                    )
                )
        return citations


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
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        sql = """
        SELECT theme_id::TEXT, theme_name::TEXT AS name, theme_description, created_at
        FROM nlp_service.themes
        WHERE theme_id = ANY(%(theme_id)s)
        """
        rows = await db.generic_read(sql, {"theme_id": [topic.id for topic in texts]})
        return [
            CitationOutput(
                id=row["theme_id"],
                citation_type=CitationType.TEXT,
                name="Theme: " + row["name"],
                summary=row["theme_description"],
                published_at=row["created_at"],
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
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        sql = """
        SELECT development_id::TEXT, label::TEXT, description, development_time
        FROM nlp_service.theme_developments
        WHERE development_id = ANY(%(development_id)s)
        """
        rows = await db.generic_read(sql, {"development_id": [topic.id for topic in texts]})
        return [
            CitationOutput(
                id=row["development_id"],
                citation_type=CitationType.TEXT,
                name="News Development: " + row["label"],
                summary=row["description"],
                published_at=row["development_time"],
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
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        sql = """
        SELECT news_id::TEXT, url, domain_url, headline, published_at
        FROM nlp_service.theme_news tn
        JOIN nlp_service.news_sources ns ON ns.source_id = tn.source_id
        WHERE tn.news_id = ANY(%(news_ids)s)
        """
        params = {"news_ids": [text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        return [
            CitationOutput(
                id=row["news_id"],
                citation_type=CitationType.LINK,
                name=row["domain_url"],
                link=row["url"],
                summary=row["headline"],
                published_at=row["published_at"],
            )
            for row in rows
        ]


@io_type
class StockEarningsSummaryText(StockText):
    id: str
    text_type: ClassVar[str] = "Earnings Call Summary"

    @classmethod
    async def _get_strs_lookup(
        cls, earnings_summaries: List[StockEarningsSummaryText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        sql = """
        SELECT summary_id::TEXT, summary
        FROM nlp_service.earnings_call_summaries
        WHERE summary_id = ANY(%(earnings_ids)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(
            sql, {"earnings_ids": [summary.id for summary in earnings_summaries]}
        )
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
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        sql = """
        SELECT created_timestamp FROM
        nlp_service.earnings_call_summaries
        WHERE summary_id = ANY(%(earnings_ids)s)
        """
        params = {"earnings_ids": [text.id for text in texts]}
        rows = await db.generic_read(sql, params)
        # TODO enhance this
        return [
            CitationOutput(
                citation_type=CitationType.TEXT,
                name=cls.text_type,
                published_at=row["created_timestamp"],
            )
            for row in rows
        ]


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
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        str_lookup = await cls._get_strs_lookup(texts)

        outputs = []
        for text in texts:
            point_str = str_lookup.get(text.id)
            if point_str:
                outputs.append(
                    CitationOutput(
                        id=text.summary_id,
                        citation_type=CitationType.TEXT,
                        name=cls.text_type,
                        summary=f"{text.summary_type}: {point_str}",
                    )
                )

        return outputs


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
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        # TODO
        return [
            CitationOutput(citation_type=CitationType.TEXT, name=text.text_type) for text in texts
        ]


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

    @classmethod
    async def _get_strs_lookup(
        cls, sec_filing_list: List[StockSecFilingText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        output: Dict[TextIDType, str] = {}
        for obj in sec_filing_list:
            filing_info = json.loads(obj.id)
            management_section = SecFiling.download_10k_10q_section(
                filing_info, section=MANAGEMENT_SECTION
            )
            risk_factor_section = SecFiling.download_10k_10q_section(
                filing_info, section=RISK_FACTORS
            )

            text = f"Management Section:\n\n{management_section}\n\nRisk Factors Section:\n\n{risk_factor_section}"  # noqa
            output[obj.id] = text

        return output

    @classmethod
    async def get_citations_for_output(
        cls, texts: List[Self], db: BoostedPG
    ) -> List[CitationOutput]:
        # TODO
        return [
            CitationOutput(citation_type=CitationType.TEXT, name=text.text_type) for text in texts
        ]


@io_type
class StockOtherSecFilingText(StockText):
    """
    Unlike `SecFilingText`, this class is used to get other types of SEC filings and the helper
    method `get_strs_lookup` is used to download the full content of the filing instead of only
    extracting a few sections.
    TODO: We may later merge this class with `SecFilingText` (and also extract certain sections
    for other types)
    """

    id: str

    text_type: ClassVar[str] = "SEC filing"

    @classmethod
    async def _get_strs_lookup(
        cls, sec_filing_list: List[StockOtherSecFilingText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        # TODO: The outputs should be cached on-demand and not downloaded every time

        output: Dict[TextIDType, str] = {}
        for obj in sec_filing_list:
            filing_info = json.loads(obj.id)
            full_content = SecFiling.download_filing_full_content(filing_info[LINK_TO_HTML])
            output[obj.id] = full_content

        return output


@io_type
class KPIText(Text):
    id: int
    text_type: ClassVar[str] = "KPI"


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
            ]
        )

    def get_citations(self, citation_ids: List[int]) -> List[Citation]:
        # do int(i) just in case GPT screwed up
        return [TextCitation(source_text=self.val[int(i)]) for i in citation_ids]

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

    def __str__(self) -> str:
        return self.val


@io_type
class TextCitation(Citation):
    source_text: Text

    @classmethod
    async def resolve_citations(cls, citations: List[Self], db: BoostedPG) -> List[CitationOutput]:
        # First group citations based on type of the source text.
        text_type_map = defaultdict(list)
        for cit in citations:
            text_type_map[type(cit.source_text)].append(cit.source_text)

        tasks = [
            typ.get_citations_for_output(text_list, db) for typ, text_list in text_type_map.items()
        ]
        # List of lists, where each nested list has outputs for each type
        outputs_nested = await gather_with_concurrency(tasks)
        # Unnest into a single list
        outputs = list(chain(*outputs_nested))

        return outputs
