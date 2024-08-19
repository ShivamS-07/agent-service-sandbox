import enum
import json
import re
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Tuple

from agent_service.io_type_utils import IO_TYPE_NAME_KEY, SerializeableBase, io_type
from agent_service.io_types.stock import StockID
from agent_service.utils.boosted_pg import BoostedPG


class TextObjectType(str, enum.Enum):
    STOCK = "stock"
    CITATION = "citation"


@io_type
class TextObject(SerializeableBase):
    """
    Represents an 'object' in a text.
    """

    type: TextObjectType

    # Index into the text
    index: int

    # For text objects that are 'link-like', store an end index. The text
    # between the start and end index will be "converted" into the object. Note
    # that this index is INCLUSIVE, and so is the final index of the string that
    # should be included in the text object.
    end_index: Optional[int] = None

    @staticmethod
    def _render(obj: "TextObject", replaced_text: Optional[str] = None) -> str:
        json_dict = obj.model_dump(mode="json")
        # Get rid of keys we don't need in the output
        json_dict.pop("index")
        json_dict.pop("end_index")
        json_dict.pop(IO_TYPE_NAME_KEY)
        if replaced_text:
            json_dict["text"] = replaced_text
        json_str = json.dumps(json_dict)
        return f" ```{json_str}``` "

    @staticmethod
    def render_text_objects(text: str, objects: List["TextObject"]) -> str:
        """
        Given a text and a list of text objects, this function 'renders' the
        text objects by inserting them into the text as markdown codeblock json
        things. (The frontend has special handling for code blocks such that
        json embedded in code blocks is parsed specially.)

        E.g.

        Input text: 'this is a text'
        Input objects: CitationTextObject(idnex=3)

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
                    output_buffer.append(TextObject._render(obj))
                i += 1

            elif i in index_replacement_object_map:
                obj = index_replacement_object_map[i]
                # We know it's not None here
                assert obj.end_index is not None
                # Output the text as a replacement for the regular text, and skip ahead
                replaced_text = text[i : obj.end_index + 1]
                output_buffer.append(TextObject._render(obj, replaced_text=replaced_text))
                i = obj.end_index + 1

        return "".join(output_buffer)


@io_type
class StockTextObject(TextObject):
    type: TextObjectType = TextObjectType.STOCK
    gbi_id: int
    symbol: Optional[str]
    company_name: Optional[str]

    @staticmethod
    async def find_stock_references_in_text(
        text: str, stocks: List[StockID], db: Optional[BoostedPG] = None
    ) -> List["StockTextObject"]:
        # First find any alternate names for the stocks
        gbi_ids = [stock.gbi_id for stock in stocks]
        if not db:
            from agent_service.utils.postgres import SyncBoostedPG

            db = SyncBoostedPG()

        sql = """
        SELECT gbi_security_id AS gbi_id, symbol, name AS company_name,
              ARRAY_AGG(an.alt_name) AS gbi_alt_names, ARRAY_AGG(can.alt_name) AS company_alt_names
        FROM master_security ms
        JOIN spiq_security_mapping ssm ON ssm.gbi_id = ms.gbi_security_id
        LEFT JOIN "data".gbi_id_alt_names an ON ms.gbi_security_id = an.gbi_id
        LEFT JOIN "data".company_alt_names can ON ssm.spiq_company_id = can.spiq_company_id
        WHERE an.enabled AND an.enddate >= NOW() AND can.enabled AND can.enddate >= NOW()
        AND ms.gbi_security_id = ANY(%(gbi_ids)s)
        GROUP BY gbi_security_id
        """

        rows = await db.generic_read(sql, params={"gbi_ids": gbi_ids})
        if not rows:
            return []

        # Maps match name to (gbi_id, symbol, company name)
        name_stock_matcher: Dict[str, Tuple[int, str, str]] = {}

        for row in rows:
            row_stock_data = (row["gbi_id"], row["symbol"], row["company_name"])
            for match_name in chain(
                [row["symbol"], row["company_name"]],  # match on symbol, company name
                row["gbi_alt_names"],  # alt names for gbi
                row["company_alt_names"],  # alt names for company
            ):
                name_stock_matcher[match_name] = row_stock_data

        # Match from longest to shortest
        matching_names_list = sorted(
            name_stock_matcher.keys(), reverse=True, key=lambda name: len(name)
        )
        regex_str = "|".join((re.escape(name) for name in matching_names_list))
        # Match any string preceded by a space and followed by space or punctuation.
        regex = re.compile(f"\\s+({regex_str})\\W+")
        seen_indexes = set()
        output = []
        for m in re.finditer(regex, text):
            # Group 1 is the match without the spaces on either side
            stock_data = name_stock_matcher.get(m.group(1))
            if not stock_data:
                continue

            index = m.start(1)
            # If we've already seen a stock reference at the same index, ignore
            # it. We're going longest to shortest so "AAPL Inc." beats "AAPL".
            if index not in seen_indexes:
                seen_indexes.add(index)
                output.append(
                    StockTextObject(
                        index=m.start(1),
                        # Our indexes are inclusive
                        end_index=m.end(1) - 1,
                        gbi_id=stock_data[0],
                        symbol=stock_data[1],
                        company_name=stock_data[2],
                    )
                )
        return output


@io_type
class CitationTextObject(TextObject):
    type: TextObjectType = TextObjectType.CITATION
    citation_id: str
