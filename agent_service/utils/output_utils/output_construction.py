from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Union

from agent_service.io_type_utils import (
    ComplexIOBase,
    HistoryEntry,
    IOType,
    TableColumnType,
    io_type,
    split_io_type_into_components,
)
from agent_service.io_types.idea import Idea
from agent_service.io_types.output import Output
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import (
    StockTableColumn,
    Table,
    TableColumn,
    TableColumnMetadata,
)
from agent_service.io_types.text import StockText, Text, TextCitation
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.output_utils.utils import io_type_to_gpt_input


def prepare_list_of_stocks(stocks: List[StockID]) -> Table:
    columns: List[TableColumn] = [StockTableColumn(data=stocks)]
    return Table(columns=columns)


def prepare_list_of_ideas(ideas: List[Idea]) -> Table:
    titles = [idea.title for idea in ideas]
    descriptions = [idea.description for idea in ideas]
    columns: List[TableColumn] = []
    columns.append(
        TableColumn(
            metadata=TableColumnMetadata(label="Name", col_type=TableColumnType.STRING), data=titles  # type: ignore
        )
    )
    columns.append(
        TableColumn(
            metadata=TableColumnMetadata(label="Description", col_type=TableColumnType.STRING),
            data=descriptions,  # type: ignore
        )
    )
    return Table(columns=columns)


def combine_text_list(
    texts: Union[List[Text], List[StockText]],
    per_line_prefix: str = "- ",
    per_line_suffix: str = "\n",
    overall_prefix: str = "",
    overall_suffix: str = "",
) -> Text:
    strings = [overall_prefix]
    all_citations = []
    cur_offset = len(overall_prefix)
    for text in texts:
        if text.title:  # output of per idea summarize, can't use -
            line = text.val + "\n\n"
        else:
            line = f"{per_line_prefix}{text.val}{per_line_suffix}"
        citations = text.get_all_citations()
        cur_offset += len(per_line_prefix)
        for cit in citations:
            if isinstance(cit, TextCitation):
                cit.citation_text_offset = (
                    cit.citation_text_offset + cur_offset if cit.citation_text_offset else None
                )
            all_citations.append(cit)
        cur_offset += len(text.val) + len(per_line_suffix)
        strings.append(line)

    entry = HistoryEntry(citations=all_citations)
    strings.append(overall_suffix)
    return Text(val="".join(strings), history=[entry])


async def prepare_list_of_texts(texts: List[Text]) -> Text:
    # First make sure all texts are resolved by fetching their data
    text_strs = await Text.get_all_strs(texts)
    for text, text_str in zip(texts, text_strs):
        text.val = text_str
        if text_str and not text.get_all_citations():
            # If a text does not have citations, add a citation to itself at the
            # end of the text. This is useful for e.g. lists of articles.
            # Need to copy to prevent circular references
            cited_text = deepcopy(text)
            cited_text.val = ""
            text.history.append(
                HistoryEntry(
                    citations=[
                        TextCitation(source_text=cited_text, citation_text_offset=len(text_str) - 1)
                    ]
                )
            )
    return combine_text_list(texts=texts)


async def prepare_list_of_stock_texts(texts: List[StockText]) -> Text:
    # Maps stocks to strings
    stock_text_map: Dict[StockID, List[StockText]] = defaultdict(list)
    text_strs = await StockText.get_all_strs(texts)
    for text, text_str in zip(texts, text_strs):
        if text.stock_id:
            # Make sure the value is set
            text.val = text_str
            # If a text does not have citations, add a citation to itself at the
            # end of the text. This is useful for e.g. lists of articles.
            if text_str and not text.get_all_citations():
                cited_text = deepcopy(text)
                cited_text.val = ""
                text.history.append(
                    HistoryEntry(
                        citations=[
                            TextCitation(
                                source_text=cited_text, citation_text_offset=len(text_str) - 1
                            )
                        ]
                    )
                )
            stock_text_map[text.stock_id].append(text)

    # Now for each stock, construct a text and stick it in a list
    stock_texts = []
    for stock, text_list in stock_text_map.items():
        prefix = f"## {stock.to_markdown_string()}\n{stock.history_to_str_with_text_objects()}\n"
        combined_text = combine_text_list(text_list, overall_prefix=prefix)
        stock_texts.append(combined_text)

    # Merge the markdown strings together
    return combine_text_list(texts=stock_texts, per_line_suffix="\n\n", per_line_prefix="")


@async_perf_logger
async def get_output_from_io_type(val: IOType, pg: BoostedPG, title: str = "") -> Output:
    """
    This function accepts any IOType and returns a 'nice' output for the
    frontend. There are special cases that are handled specifically, otherwise
    we do our best.
    """
    # Be a bit sneaky, check if there's a title attached to the object
    if not title and hasattr(val, "title"):
        title = str(val.title)

    if isinstance(val, list):
        if not val:
            # TODO probably improve this
            val = Text(val="No values found.")
        elif isinstance(val[0], StockID):
            val = prepare_list_of_stocks(stocks=val)
        elif isinstance(val[0], StockText):
            val = await prepare_list_of_stock_texts(texts=val)
        elif isinstance(val[0], Text):
            val = await prepare_list_of_texts(texts=val)
        elif isinstance(val[0], Idea):
            val = prepare_list_of_ideas(ideas=val)
        else:
            coros = [get_output_from_io_type(v, pg=pg) for v in val]
            val = await gather_with_concurrency(coros, n=len(coros))
    elif isinstance(val, dict):
        # Ideally we shouldn't ever hit this, tools should not output raw dicts.
        val = {key: await get_output_from_io_type(v, pg=pg) for key, v in val.items()}
    if not isinstance(val, ComplexIOBase):
        val = Text.from_io_type(val)
    output_val = await val.to_rich_output(pg, title=title)
    if output_val.citations:
        output_val.citations = sorted(
            output_val.citations, key=lambda cit: (cit.inline_offset or 0)
        )

    return output_val


@io_type
class PreparedOutput(ComplexIOBase):
    """Wrapper type around ANY IO type that includes a title."""

    val: IOType
    title: str = ""

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        return await get_output_from_io_type(val=self.val, pg=pg, title=self.title)

    async def to_gpt_input(self, use_abbreviated_output: bool = True) -> str:
        return await io_type_to_gpt_input(self.val, use_abbreviated_output=use_abbreviated_output)

    async def split_into_components(self, main_title: Optional[str] = None) -> List["IOType"]:
        split_vals = await split_io_type_into_components(self.val, main_title=self.title)
        if len(split_vals) == 1:
            return [PreparedOutput(val=split_vals[0], title=self.title)]

        return [PreparedOutput(val=new_val, title=new_val.title) for new_val in split_vals]  # type: ignore


# TODO remove me, for backwards compat
@io_type
class TitledIOType(PreparedOutput):
    pass
