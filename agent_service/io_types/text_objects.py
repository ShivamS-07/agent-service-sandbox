import enum
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional

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
                for obj in text_objects:
                    output_buffer.append(TextObject.render_object_to_json(obj))
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
        original_text: str, tagged_text: str, symbol_to_stock_map: Dict[str, StockID], db: BoostedPG
    ) -> List:
        # It's a nightmare regex, but really it's just:
        # match two open brackets, match any number of non-close bracket characters, match two close brackets
        tag_regex = re.compile(r"\[\[([^\]]*)\]\]")
        text_objects: List[StockTextObject] = []
        for tag_match in re.finditer(tag_regex, tagged_text):
            if not tag_match or not tag_match.group(1):
                continue
            symbol = tag_match.group(1)
            if symbol not in symbol_to_stock_map:
                continue

            # The below code is necessary because GPT is a nightmare and keeps
            # inserting random whitespace that messes with the indexes matching
            # up with the original text.

            # NOTE: this is likely extremely slow, but for now it works. This
            # is done during the workflow, so probably not a huge deal.
            # Now that we have a symbol, we want to find the place that symbol
            # occurs in the original text, so that we can fill in the stock
            # object.
            original_text_stock_references = list(re.finditer(symbol, original_text))
            tagged_text_stock_references = list(re.finditer(symbol, tagged_text))
            if len(original_text_stock_references) != len(tagged_text_stock_references):
                continue
            for i, symbol_match in enumerate(tagged_text_stock_references):
                # The symbol match matches the first group of the tag match
                # (i.e. where the symbol starts after the brackets)
                if symbol_match.start() == tag_match.start(1):
                    stock = symbol_to_stock_map[symbol]
                    original_text_location = original_text_stock_references[i]

                    text_objects.append(
                        StockTextObject(
                            gbi_id=stock.gbi_id,
                            symbol=stock.symbol,
                            company_name=stock.company_name,
                            index=original_text_location.start(),
                            end_index=original_text_location.end() - 1,
                            isin="",
                        )
                    )
                    break

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
    ) -> List["TextObject"]:
        if not context.stock_info:
            return []
        if not db:
            from agent_service.utils.postgres import SyncBoostedPG

            db = SyncBoostedPG()
        symbol_to_stock_map = {}
        for stock in context.stock_info:
            if stock.symbol:
                symbol_to_stock_map[stock.symbol] = stock
            if stock.company_name:
                symbol_to_stock_map[stock.company_name] = stock

        gpt_context = create_gpt_context(
            GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
        )
        llm = GPT(context=gpt_context, model=GPT4_O_MINI)

        object_tagging_prompt = Prompt(
            name="AGENT_SVC_TEXT_TAGGING_POSTPROCESSING",
            template="""
You are a financial analyst reading a piece of text from a report or
summary. The text is written in markdown. Your job is to tag each stock or
company mention with wiki link style double brackets. For example, the mention
of "AAPL" should become "[[AAPL]]". These mentions will be highlighted for your
clients. If the ticker of a company follows the name like "Apple Inc. (AAPL)"
you should ONLY tag the ticker: "Apple Inc. ([[AAPL]])". Other than that, make
NO changes to the input text AT ALL. If you do, you will be fired!
{chat_context} Here is the text that you should perform the tagging on: {text}

Your tagged output text:""",
        )

        chat_context = ""
        if context.chat:
            chat_context = (
                "For context, the following is the chat exchange between"
                f" you and your client:\n{context.chat.get_gpt_input()}"
            )

        result = await llm.do_chat_w_sys_prompt(
            main_prompt=object_tagging_prompt.format(text=text, chat_context=chat_context),
            sys_prompt=NO_PROMPT,
        )

        return await TextObject._extract_stock_tags_from_text(
            original_text=text, tagged_text=result, symbol_to_stock_map=symbol_to_stock_map, db=db
        )


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
