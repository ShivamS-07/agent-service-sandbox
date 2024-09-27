import enum
import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, cast

from agent_service.GPT.constants import GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import (
    IO_TYPE_NAME_KEY,
    PrimitiveType,
    ScoreOutput,
    SerializeableBase,
    TableColumnType,
    io_type,
)
from agent_service.io_types.stock import StockID
from agent_service.types import PlanRunContext
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.stock_metadata import StockMetadata, get_stock_metadata

logger = logging.getLogger(__name__)


class TextObjectType(enum.StrEnum):
    STOCK = "stock"
    CITATION = "citation"


@io_type
class TextObject(SerializeableBase):
    """
    Represents an 'object' in a text.
    """

    type: TextObjectType | TableColumnType

    # Index into the text
    index: int

    # For text objects that are 'link-like', store an end index. The text
    # between the start and end index will be "converted" into the object. Note
    # that this index is INCLUSIVE, and so is the final index of the string that
    # should be included in the text object.
    end_index: Optional[int] = None

    @staticmethod
    def render_object_to_json(obj: "TextObject", replaced_text: Optional[str] = None) -> str:
        json_dict = obj.model_dump(mode="json")
        # Get rid of keys we don't need in the output
        json_dict.pop("index")
        json_dict.pop("end_index")
        json_dict.pop(IO_TYPE_NAME_KEY)
        if replaced_text:
            json_dict["text"] = replaced_text
        json_str = json.dumps(json_dict)
        return f"```{json_str}```"

    @staticmethod
    def render_text_objects(text: str, objects: List["TextObject"]) -> str:
        """
        Given a text and a list of text objects, this function 'renders' the
        text objects by inserting them into the text as markdown codeblock json
        things. (The frontend has special handling for code blocks such that
        json embedded in code blocks is parsed specially.)

        E.g.

        Input text: 'this is a text'
        Input objects: CitationTextObject(index=3)

        Output text: 'this```{"type": "citation", "citation_id": "..."}``` is a text
        """
        if not objects:
            return text

        index_object_map: Dict[int, List[TextObject]] = defaultdict(list)
        # A "replacement" object is one that replaces text in the output, and
        # doesn't just add to it. These are `like-like` things that are inserted
        # to replace a range of text. We only support one replacement per index
        # because nested objects wouldn't work.
        index_replacement_object_map: Dict[int, TextObject] = {}
        # Sort by the length of the span, if there is one. That way the object
        # that ends up in index_replacement_object_map is the longest span for
        # that index (in the very strange/rare case there are duplicates)
        # E.g. "Apple Inc." should be highlighted over just "Apple".
        for obj in sorted(objects, key=lambda o: (o.end_index or 0) - o.index):
            if obj.end_index:
                index_replacement_object_map[obj.index] = obj
            else:
                index_object_map[obj.index].append(obj)

        i = 0
        output_buffer = []
        while i < len(text):
            if i not in index_object_map and i not in index_replacement_object_map:
                output_buffer.append(text[i])
                i += 1
                continue

            # Replacement objects (like stock links) and normal objects (like
            # citations) shouldn't ever really conflict, but if they do we
            # prioritize the normal non-replacement object. We don't support
            # e.g. both a citation and a stock link starting at the same point.

            if i in index_object_map:
                # Simple case, just keep outputting the objects one after the other
                text_objects = index_object_map[i]
                output_buffer.append(text[i])
                object_list = []
                for obj in text_objects:
                    object_list.append(TextObject.render_object_to_json(obj))
                output_buffer.append(" ".join(object_list))
                i += 1

            elif i in index_replacement_object_map:
                obj = index_replacement_object_map[i]
                # We know it's not None here
                assert obj.end_index is not None
                # Output the text as a replacement for the regular text, and skip ahead
                replaced_text = text[i : obj.end_index + 1]
                output_buffer.append(
                    TextObject.render_object_to_json(obj, replaced_text=replaced_text)
                )
                i = obj.end_index + 1

        return "".join(output_buffer)

    @staticmethod
    async def _extract_stock_tags_from_text(
        text: str, stocks: Dict[str, StockID], db: BoostedPG
    ) -> List["StockTextObject"]:
        text_objects: List[StockTextObject] = []

        for company_name in stocks:
            symbol = cast(str, stocks[company_name].symbol)
            # Find the first occurrence of either the company name or ticker symbol
            company_match_regex = rf"\b{re.escape(company_name)}"
            symbol_match_regex = rf"\b{re.escape(symbol)}"
            company_match = re.search(company_match_regex, text)
            symbol_match = re.search(symbol_match_regex, text)

            if company_match and symbol_match:
                company_start = company_match.start()
                company_end = company_match.end()
                symbol_start = symbol_match.start()
                symbol_end = symbol_match.end()

                # Check if symbol occurs immediately after company name with only
                # spaces and parentheses in between (eg. "Apple Inc. (AAPL)")
                between_text = text[company_end:symbol_start]
                if between_text and re.fullmatch(r"[\s()]*", between_text):
                    # Use symbol's position
                    stock_location_start = symbol_start
                    stock_location_end = symbol_end - 1
                else:
                    # Use whichever occurs first
                    if company_start <= symbol_start:
                        stock_location_start = company_start
                        stock_location_end = company_end - 1
                    else:
                        stock_location_start = symbol_start
                        stock_location_end = symbol_end - 1

            elif symbol_match:
                stock_location_start = symbol_match.start()
                stock_location_end = symbol_match.end() - 1

            elif company_match:
                stock_location_start = company_match.start()
                stock_location_end = company_match.end() - 1

            else:
                continue

            text_objects.append(
                StockTextObject(
                    gbi_id=stocks[company_name].gbi_id,
                    symbol=stocks[company_name].symbol,
                    company_name=company_name,
                    index=stock_location_start,
                    end_index=stock_location_end,
                    isin="",
                )
            )

        stock_metadata = await get_stock_metadata(gbi_ids=[to.gbi_id for to in text_objects], pg=db)
        # Enrich with other data
        for text_object in text_objects:
            metadata = stock_metadata.get(text_object.gbi_id)
            if not metadata:
                continue
            text_object.isin = metadata.isin
            text_object.sector = metadata.sector
            text_object.subindustry = metadata.subindustry
            text_object.exchange = metadata.exchange

        return text_objects

    @staticmethod
    async def find_and_tag_references_in_text(
        text: str, context: PlanRunContext, db: Optional[BoostedPG] = None
    ) -> List["StockTextObject"]:
        from agent_service.tools.stocks import (  # circular import fix
            StockIdentifierLookupInput,
            stock_identifier_lookup,
        )

        if not context.stock_info:
            return []
        if not db:
            from agent_service.utils.postgres import SyncBoostedPG

            db = SyncBoostedPG()

        gpt_context = create_gpt_context(
            GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
        )
        llm = GPT(context=gpt_context, model=GPT4_O_MINI)

        object_tagging_prompt = Prompt(
            name="AGENT_SVC_TEXT_TAGGING_POSTPROCESSING",
            template="""You are a financial analyst tasked with identifying **every single**
publicly traded company mentioned in a block of text, no matter how briefly it is mentioned.
Even if the company is only mentioned once or in passing, it must be included in the final output.
Your job is to:
1. Carefully identify **every** publicly traded company, regardless of its prominence in the text.
2. Output the corresponding stock ticker symbols for each publicly traded company.
3. Your output format should be a dictionary where the keys are the company names and the
values are the ticker symbols. Each company should only appear once.

Here are additional instructions to help you perform the task:
- You must identify **all publicly traded company names** in the text and only the publicly traded
company names, whether they are the main subject or mentioned in passing.
- If the text contains a company's stock ticker symbol, use that as the value in the dictionary.
- The output should be only the dictionary containing the key-value pairs, with no additional text
or commentary. Failure to comply means you will lose your job!

This is an example output format: {{"Apple":"AAPL", "Tesla":"TSLA", "Nvidia":"NVDA"}}

The text you need to analyze is provided below.
{text}
""",
        )

        result = await llm.do_chat_w_sys_prompt(
            main_prompt=object_tagging_prompt.format(text=text),
            sys_prompt=NO_PROMPT,
        )

        symbol_to_stock_map = {}
        for stock_obj in context.stock_info:
            if stock_obj.symbol:
                symbol_to_stock_map[stock_obj.symbol] = stock_obj

        try:
            stocks: dict = json.loads(result)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse dictionary string: {result}")

        for stock_name in list(stocks.keys()):
            if stocks[stock_name] not in symbol_to_stock_map:
                try:
                    stocks[stock_name] = await stock_identifier_lookup(  # type: ignore
                        StockIdentifierLookupInput(stock_name=stock_name), context
                    )
                except ValueError:
                    # If stock cannot be found in the db, remove it from the dict
                    logger.error(f"Could not find stock for {stock_name}")
                    del stocks[stock_name]
            else:
                stocks[stock_name] = symbol_to_stock_map[stocks[stock_name]]

        return await TextObject._extract_stock_tags_from_text(text=text, stocks=stocks, db=db)


@io_type
class StockTextObject(TextObject, StockMetadata):
    type: TextObjectType | TableColumnType = TextObjectType.STOCK


@io_type
class CitationTextObject(TextObject):
    type: TextObjectType | TableColumnType = TextObjectType.CITATION
    citation_id: str


@io_type
class BasicTextObject(TextObject):
    """
    Represents a text object for types that are created on the fly and handled on
    the frontend. E.g. floats, percents, etc.
    """

    type: TextObjectType | TableColumnType
    value: PrimitiveType | ScoreOutput
    unit: Optional[str] = None
