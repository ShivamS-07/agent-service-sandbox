from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Type, Union
from uuid import uuid4

import pandas as pd
from pydantic import Field
from pydantic.functional_serializers import field_serializer
from pydantic.functional_validators import field_validator

from agent_service.io_type_utils import ComplexIOBase, io_type


@io_type
class Table(ComplexIOBase):
    # A dataframe wrapper
    val: pd.DataFrame

    def to_gpt_input(self) -> str:
        return f"[Table with {self.val.shape[0]} rows and {self.val.shape[0]} columns]"

    @field_validator("val", mode="before")
    @classmethod
    def _deserializer(cls, val: Any) -> Any:
        val_field = cls.model_fields["val"]
        if isinstance(val, dict) and val_field.annotation is pd.DataFrame:
            val = pd.DataFrame.from_dict(val)
        return val

    @field_serializer("val", mode="wrap")
    @classmethod
    def _field_serializer(cls, val: Any, dumper: Callable) -> Any:
        if isinstance(val, pd.DataFrame):
            val = val.to_dict()
        return dumper(val)


@io_type
class TimeSeriesTable(Table):
    # A dataframe with date row index.
    val: pd.DataFrame


@io_type
class StockTimeSeriesTable(TimeSeriesTable):
    # A dataframe with date row index and GBI ID columns.
    pass


@io_type
class StockTable(Table):
    # A dataframe with GBI ID row index and arbitrary columns.
    pass


@io_type
class Graph(ComplexIOBase):
    # TODO: figure out how we are going to represent a graph, now just a table
    val: Table

    def to_gpt_input(self) -> str:
        return f"[Graph based on table with {self.val.val.shape[0]} rows and {self.val.val.shape[0]} columns]"


@io_type
class TimeSeriesLineGraph(Graph):
    pass


TextIDType = Union[str, int]


@io_type
class Text(ComplexIOBase):
    id: TextIDType = Field(default_factory=lambda: str(uuid4()))
    val: str = ""

    @classmethod
    def get_all_strs(cls, text: Union[Text, List[Any]]) -> Union[str, List[Any]]:
        # For any possible configuration of Texts in Lists, this converts that
        # configuration to the corresponding strings in list, in class specific batches

        def convert_to_ids_and_categorize(
            text: Union[Text, List[Any]], categories: Dict[Type[Text], List[Text]]
        ) -> Union[TextIDType, List[Any]]:
            """
            Convert a structure of texts (e.g. nested lists, etc.) into an
            identical structure of text ID's, while also keeping track of all
            texts of each type.
            """
            if not isinstance(text, list):
                categories[type(text)].append(text)
                return text.id
            else:
                return [convert_to_ids_and_categorize(sub_text, categories) for sub_text in text]

        categories: Dict[Type[Text], List[Text]] = defaultdict(list)
        # identical structure to input texts, but as IDs
        texts_as_ids = convert_to_ids_and_categorize(text, categories)
        strs_lookup = {}
        # For every subclass of Text, fetch data
        for textclass, texts in categories.items():
            strs_lookup.update(textclass.get_strs_lookup(texts))

        def convert_ids_to_strs(
            strs_lookup: Dict[TextIDType, str], id_rep: Union[TextIDType, List[Any]]
        ) -> Union[str, List[Any]]:
            """
            Take the structure of ID lists, and map back into actual strings.
            """
            if isinstance(id_rep, list):
                return [convert_ids_to_strs(strs_lookup, sub_id_rep) for sub_id_rep in id_rep]
            else:
                return strs_lookup[id_rep]

        return convert_ids_to_strs(strs_lookup, texts_as_ids)

    @classmethod
    def get_strs_lookup(cls, texts: List[Text]) -> Dict[TextIDType, str]:
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
class StockNewsDevelopmentText(Text):
    id: str

    @classmethod
    def get_strs_lookup(cls, news_topics: List[StockNewsDevelopmentText]) -> Dict[TextIDType, str]:  # type: ignore
        sql = """
        SELECT topic_id::TEXT, (topic_descriptions->-1->0)::TEXT AS description
        FROM nlp_service.stock_news_topics
        WHERE topic_id = ANY(%(topic_ids)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"topic_ids": [topic.id for topic in news_topics]})
        return {row["topic_id"]: row["description"] for row in rows}


@io_type
class ThemeText(Text):
    id: str
    val: Any = None

    @classmethod
    def get_strs_lookup(cls, themes: List[ThemeText]) -> Dict[str, str]:  # type: ignore
        sql = """
        SELECT theme_id::TEXT, theme_descriptions::TEXT AS description
        FROM nlp_service.themes
        WHERE theme_id = ANY(%(theme_id)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"theme_id": [topic.id for topic in themes]})
        return {row["theme_id"]: row["description"] for row in rows}


@io_type
class ThemeNewsDevelopmentText(Text):
    id: str
    val: Any = None

    @classmethod
    def get_strs_lookup(cls, themes: List[ThemeNewsDevelopmentText]) -> Dict[str, str]:  # type: ignore
        sql = """
        SELECT development_id::TEXT, label::TEXT, description::TEXT
        FROM nlp_service.theme_developments
        WHERE development_id = ANY(%(development_id)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"development_id": [topic.id for topic in themes]})
        return {row["development_id"]: row["label"] + "\n" + row["description"] for row in rows}


@io_type
class ThemeNewsDevelopmentArticlesText(Text):
    id: str
    val: Any = None

    @classmethod
    def get_strs_lookup(cls, developments: List[ThemeNewsDevelopmentArticlesText]) -> Dict[str, str]:  # type: ignore
        sql = """
        SELECT news_id::TEXT, headline::TEXT, summary::TEXT
        FROM nlp_service.theme_news
        WHERE news_id = ANY(%(news_id)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"news_id": [topic.id for topic in developments]})
        return {row["news_id"]: row["headline"] + "\n" + row["summary"] for row in rows}


@io_type
class EarningsSummaryText(Text):
    @classmethod
    def get_strs_lookup(
        cls, earnings_summaries: List[EarningsSummaryText]  # type: ignore
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
class CompanyDescriptionText(Text):
    id: int  # gbi_id

    @classmethod
    def get_strs_lookup(
        cls, company_descriptions: List[CompanyDescriptionText]  # type: ignore
    ) -> Dict[TextIDType, str]:
        sql = """
        SELECT ssm.gbi_id, cds.company_description_short
        FROM spiq_security_mapping ssm
        JOIN nlp_service.company_descriptions_short cds
        ON cds.spiq_company_id = ssm.spiq_company_id
        WHERE ssm.gbi_id = ANY(%(stocks)s)
        """
        from agent_service.utils.postgres import get_psql

        db = get_psql()
        rows = db.generic_read(sql, {"stocks": [desc.id for desc in company_descriptions]})
        return {row["gbi_id"]: row["company_description_short"] for row in rows}
