from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

import mdutils
from pydantic import Field

from agent_service.external.sec_utils import MANAGEMENT_SECTION, RISK_FACTORS, SecFiling
from agent_service.io_type_utils import ComplexIOBase, IOType, io_type
from agent_service.io_types.output import Output, OutputType
from agent_service.utils.boosted_pg import BoostedPG

TextIDType = Union[str, int]


@io_type
class Text(ComplexIOBase):
    id: TextIDType = Field(default_factory=lambda: str(uuid4()))
    val: str = ""
    text_type: ClassVar[str] = "Misc"
    gbi_id: Optional[int] = None

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Text):
            return self.id == other.id
        return False

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        return TextOutput(val=self.get().val, title=title)

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
    def get_all_strs(
        cls, text: Union[Text, TextGroup, List[Any], Dict[Any, Any]], include_header: bool = False
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
        strs_lookup = {}
        # For every subclass of Text, fetch data
        for textclass, texts in categories.items():
            strs_lookup.update(textclass.get_strs_lookup(texts, include_header=include_header))

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
                return id_rep.convert_to_str(strs_lookup)
            else:
                return strs_lookup[id_rep]

        return convert_ids_to_strs(strs_lookup, texts_as_ids)

    @classmethod
    def get_strs_lookup(
        cls, texts: List[Text], include_header: bool = False
    ) -> Dict[TextIDType, str]:
        strs_lookup = cls._get_strs_lookup(texts)
        if include_header:
            for id, val in strs_lookup.items():
                strs_lookup[id] = f"Text type: {cls.text_type}\nText:\n{val}"
        return strs_lookup

    @classmethod
    def _get_strs_lookup(cls, texts: List[Text]) -> Dict[TextIDType, str]:
        return {text.id: text.val for text in texts}

    def get(self) -> Text:
        """
        For an instance of a 'Text' subclass, resolve and return it as a standard Text object.
        """
        if not self.val:
            # resolve the text if necessary
            lookup = self.get_strs_lookup([self])
            return Text(val=lookup[self.id])
        else:
            return Text(val=self.val)


@io_type
class NewsText(Text):
    pass


@io_type
class StockText(Text):
    pass


@io_type
class StockNewsDevelopmentText(NewsText, StockText):
    id: str
    text_type: ClassVar[str] = "News Development Summary"

    @classmethod
    def _get_strs_lookup(cls, news_topics: List[StockNewsDevelopmentText]) -> Dict[TextIDType, str]:  # type: ignore
        sql = """
        SELECT topic_id::TEXT, topic_label, (topic_descriptions->-1->0)::TEXT AS description
        FROM nlp_service.stock_news_topics
        WHERE topic_id = ANY(%(topic_ids)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"topic_ids": [topic.id for topic in news_topics]})
        return {row["topic_id"]: f"{row['topic_label']}\n{row['description']}" for row in rows}


@io_type
class StockNewsDevelopmentArticlesText(NewsText, StockText):
    id: str
    text_type: ClassVar[str] = "News Article Summary"

    @classmethod
    def _get_strs_lookup(
        cls, news_topics: List[StockNewsDevelopmentArticlesText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        from agent_service.utils.postgres import get_psql

        sql = """
            SELECT news_id::TEXT, headline, summary
            FROM nlp_service.stock_news
            WHERE news_id = ANY(%(news_ids)s)
        """
        rows = get_psql().generic_read(sql, {"news_ids": [topic.id for topic in news_topics]})
        return {row["news_id"]: f"{row['headline']}\n{row['summary']}" for row in rows}


@io_type
class NewsPoolArticleText(NewsText):
    id: str
    text_type: ClassVar[str] = "News Article Summary"

    @classmethod
    def _get_strs_lookup(cls, news_pool: List[NewsPoolArticleText]) -> Dict[str, str]:  # type: ignore
        sql = """
        SELECT news_id::TEXT, headline::TEXT, summary::TEXT
        FROM nlp_service.news_pool
        WHERE news_id = ANY(%(news_ids)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"news_ids": [topic.id for topic in news_pool]})
        return {row["news_id"]: f"{row['headline']}\n{row['summary']}" for row in rows}


@io_type
class ThemeText(Text):
    id: str
    text_type: ClassVar[str] = "Theme Description"

    @classmethod
    def _get_strs_lookup(cls, themes: List[ThemeText]) -> Dict[str, str]:  # type: ignore
        sql = """
        SELECT theme_id::TEXT, theme_description::TEXT AS description
        FROM nlp_service.themes
        WHERE theme_id = ANY(%(theme_id)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"theme_id": [topic.id for topic in themes]})
        return {row["theme_id"]: row["description"] for row in rows}


@io_type
class ThemeNewsDevelopmentText(NewsText):
    id: str
    text_type: ClassVar[str] = "News Development Summary"

    @classmethod
    def _get_strs_lookup(cls, themes: List[ThemeNewsDevelopmentText]) -> Dict[str, str]:  # type: ignore
        sql = """
        SELECT development_id::TEXT, label::TEXT, description::TEXT
        FROM nlp_service.theme_developments
        WHERE development_id = ANY(%(development_id)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"development_id": [topic.id for topic in themes]})
        return {row["development_id"]: f"{row['label']}\n{row['description']}" for row in rows}


@io_type
class ThemeNewsDevelopmentArticlesText(NewsText):
    id: str
    text_type: ClassVar[str] = "News Article Summary"

    @classmethod
    def _get_strs_lookup(cls, developments: List[ThemeNewsDevelopmentArticlesText]) -> Dict[str, str]:  # type: ignore
        sql = """
        SELECT news_id::TEXT, headline::TEXT, summary::TEXT
        FROM nlp_service.theme_news
        WHERE news_id = ANY(%(news_id)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"news_id": [topic.id for topic in developments]})
        return {row["news_id"]: f"{row['headline']}\n{row['summary']}" for row in rows}


@io_type
class StockEarningsSummaryText(StockText):
    id: str
    text_type: ClassVar[str] = "Earnings Call Summary"

    @classmethod
    def _get_strs_lookup(
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


@io_type
class StockDescriptionText(StockText):
    id: int  # gbi_id
    text_type: ClassVar[str] = "Company Description"

    @classmethod
    def _get_strs_lookup(
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

        return descriptions


@io_type
class StockSecFilingText(StockText):
    """
    The ID field is a serialized JSON object containing the latest SEC filing information.
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
    def _get_strs_lookup(
        cls, sec_filing_list: List[StockSecFilingText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        output: Dict[TextIDType, str] = {}
        for obj in sec_filing_list:
            filing_info = json.loads(obj.id)
            management_section = SecFiling.download_section(filing_info, section=MANAGEMENT_SECTION)
            risk_factor_section = SecFiling.download_section(filing_info, section=RISK_FACTORS)

            text = f"Management Section:\n\n{management_section}\n\nRisk Factors Section:\n\n{risk_factor_section}"  # noqa
            output[obj.id] = text

        return output


@io_type
class KPIText(Text):
    id: int
    text_type: ClassVar[str] = "KPI"


# These are not actual Text types, but build on top of them


@io_type
class TextGroup(ComplexIOBase):
    val: List[StockText]

    @staticmethod
    def join(group1: TextGroup, group2: TextGroup) -> TextGroup:
        return TextGroup(val=group1.val + group2.val)

    def convert_to_str(self, id_to_str: Dict[TextIDType, str]) -> str:
        return "\n***\n".join(id_to_str[text.id] for text in self.val)

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        from agent_service.utils.output_construction import get_output_from_io_type

        # construct a lookup for all child texts
        strings = Text.get_all_strs(self.val)
        # TODO fix this implementation?
        return await get_output_from_io_type(strings, pg=pg, title=title)


@io_type
class EquivalentKPITexts(TextGroup):
    val: List[KPIText]  # type: ignore
    general_kpi_name: str


class TextOutput(Output):
    output_type: Literal[OutputType.TEXT] = OutputType.TEXT
    val: str

    def __str__(self) -> str:
        return self.val
