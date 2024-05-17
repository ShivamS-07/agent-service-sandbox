from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Type, Union

import pandas as pd
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


@io_type
class Text(ComplexIOBase):
    id: str

    @classmethod
    def get_all_strs(cls, text: Union[Text, List[Any]]) -> Union[str, List[Any]]:
        # For any possible configuration of Texts in Lists, this converts that
        # configuration to the corresponding strings in list, in class specific batches

        def convert_to_ids_and_categorize(
            text: Union[Text, List[Any]], categories: Dict[Type[Text], List[Text]]
        ) -> Union[str, List[Any]]:
            if not isinstance(text, list):
                categories[type(text)].append(text)
                return text.id
            else:
                return [convert_to_ids_and_categorize(sub_text, categories) for sub_text in text]

        categories: Dict[Type, List[Text]] = defaultdict(list)
        id_rep = convert_to_ids_and_categorize(text, categories)
        strs_lookup = {}
        for textclass, texts in categories.items():
            strs_lookup.update(textclass.get_strs_lookup(texts))

        def convert_ids_to_strs(
            strs_lookup: Dict[str, str], id_rep: Union[str, List[Any]]
        ) -> Union[str, List[Any]]:
            if isinstance(id_rep, str):
                return strs_lookup[id_rep]
            else:
                return [convert_ids_to_strs(strs_lookup, sub_id_rep) for sub_id_rep in id_rep]

        return convert_ids_to_strs(strs_lookup, id_rep)

    @classmethod
    @abstractmethod
    def get_strs_lookup(cls, texts: List[Text]) -> Dict[str, str]:
        pass


@io_type
class NewsDevelopmentText(Text):
    id: str

    @classmethod
    def get_strs_lookup(cls, news_topics: List[NewsDevelopmentText]) -> Dict[str, str]:  # type: ignore
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
class EarningsSummaryText(Text):
    id: str

    @classmethod
    def get_strs_lookup(cls, earnings_summaries: List[EarningsSummaryText]) -> Dict[str, str]:  # type: ignore
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
class SummaryText(Text):
    val: str

    @classmethod
    def get_strs_lookup(cls, earnings_summaries: List[SummaryText]) -> Dict[str, str]:  # type: ignore
        return {summary.id: summary.val for summary in earnings_summaries}
