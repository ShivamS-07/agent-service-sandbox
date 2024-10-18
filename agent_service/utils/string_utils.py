import json
import logging
import re
import uuid
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from json_repair import repair_json  # type: ignore

logger = logging.getLogger(__name__)

JSON_START_CHARS = "{["
JSON_END_CHARS = "}]"
JSON_END_CHARS_ESCAPED = r"\}\]"
# need re.MULTILINE and re.DOTALL since the string potentially has newlines
CLEAN_JSON_RE = re.compile(
    f"^[^{JSON_START_CHARS}]*([{JSON_START_CHARS}].*"
    f"[{JSON_END_CHARS_ESCAPED}])[^{JSON_END_CHARS_ESCAPED}]*$",
    re.MULTILINE | re.DOTALL,
)
BOLD_TAGS = ["<b>", "</b>"]
START_HEADER = "__START__"

# Precompiled regex patterns
STYLE_TAG_CLEANUP_PATTERN = re.compile(r"<style.*?>.*?</style>", re.DOTALL | re.IGNORECASE)
NEWLINE_CLEANUP_PATTERN = re.compile(r"\n\s*\n\s*\n+")


def clean_to_json_if_needed(json: str, repair: bool = True) -> str:
    if not json:
        return json

    # removes any text before or after a json object/array
    # if repair is true, tries to repair any errors inside the json
    if json[0] in JSON_START_CHARS and json[-1] in JSON_END_CHARS:
        # don't clean if already looks good
        if repair:
            json = repair_json_if_needed(json)
        return json
    else:
        match = CLEAN_JSON_RE.match(json)
        if match:
            json = match.group(1)
            if repair:
                json = repair_json_if_needed(json)
            return json
        return json  # doesn't seem to be a json, but just return


def repair_json_if_needed(json_str: str, json_load: bool = False) -> Any:
    try:
        loaded_obj = json.loads(json_str)
        return loaded_obj if json_load else json_str
    except json.JSONDecodeError as e:
        # when logging is true, it always returns object, not str
        new_cleaned_json, log = repair_json(json_str, logging=True)
        if new_cleaned_json:  # will be an empty str if failed to repair
            if log:
                logger.warning(f"Actions taken to repair JSON: {log}")
            if json_load:
                return new_cleaned_json
            else:
                return json.dumps(new_cleaned_json)

        logger.warning(
            f"Failed to repair JSON: {json_str} due to error: {e}. Returning the original string."
        )
        return json_str


def remove_bolding(S: str) -> str:
    # removes bold tags from S
    for tag in BOLD_TAGS:
        S = S.replace(tag, "")
    return S


def looks_like_title(paragraph: str) -> bool:
    # check if a paragraph looks like a title based on length and lack of punctuation
    return (
        len(paragraph) < 60
        and paragraph.replace(" ", "").replace("-", "").replace(",", " ").isalnum()
    )


def is_spaced(i: int, paragraphs: List[str]) -> bool:
    # i is the index of the current paragraph in paragraphs
    # check if a paragraph is spaced away from the rest of the text, all headers are
    # first line could be a header, last one wouldn't be
    return (i == 0 or not paragraphs[i - 1]) and (i < len(paragraphs) - 1 and not paragraphs[i - 1])


def not_in_table(i: int, paragraphs: List[str]) -> bool:
    # i is the index of the current paragraph in paragraphs
    # Paragraphs that look like headers but are actually in tables are typically surrounded by
    # other short paragraphs (with a buffer of 1 empty line, so pass other checks).
    # This function returns True if the paragraph at i has longer context around it,
    # indicating it is likely not in a table (not perfect, but mostly works)
    return (i > 1 and len(paragraphs[i - 2]) > 50) or (
        i < len(paragraphs) - 2 and len(paragraphs[i + 2]) > 50
    )


def get_sections(sec_text: str) -> Dict[str, str]:
    """
    Converts the sec_text (a text with sections) into a mapping of headers to the body of their
    section. The header is included in the body. The text appearing before the first detected header
    is given the "__START__" header.
    """
    paragraphs = sec_text.split("\n")
    curr_section_header = START_HEADER
    section_lookup: Dict[str, List[str]] = {START_HEADER: []}
    section_headers = []
    for i, paragraph in enumerate(paragraphs):
        if (
            paragraph.strip()
            and looks_like_title(paragraph)
            and is_spaced(i, paragraphs)
            and not_in_table(i, paragraphs)
        ):
            curr_section_header = paragraph.strip()
            section_lookup[curr_section_header] = [curr_section_header]
            section_headers.append(curr_section_header)
        else:
            section_lookup[curr_section_header].append(paragraph)

    final_lookup: Dict[str, str] = {}
    for header, body in section_lookup.items():
        if any(body):
            final_lookup[header] = "\n".join(body)

    return final_lookup


def paren(S: str) -> str:
    return f"({S})"


def safe_json_dump(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj)


def safe_json_load(obj: Optional[str]) -> Any:
    if obj is None:
        return None
    return json.loads(obj)


def strip_code_backticks(obj: str) -> str:
    outputs = []
    for line in obj.split("\n"):
        if line.startswith("```"):
            continue
        outputs.append(line)
    return "\n".join(outputs)


def is_valid_uuid(input: str) -> bool:
    try:
        uuid.UUID(input)
        return True
    except ValueError:
        return False


class HTMLFilter(HTMLParser):
    text = ""

    def handle_data(self, data: str) -> None:
        self.text += data


def html_to_text(html_str: str) -> str:
    """
    Transforms HTML to plain text
    First runs preprocessing, where we manually strip out the <style ... style> and <script ... script> contents
    Then parses out the rest of the HTML
    Finally runs postprocessing, reducing the number of newlines by replacing any instances of more than 2 \n with
    just \n\n
    """
    # Preprocessing - clean up the <style> tag which the parser misses consistently
    html_str = re.sub(STYLE_TAG_CLEANUP_PATTERN, "", html_str)

    # Parsing
    parser = HTMLFilter()
    parser.feed(html_str)
    plaintext_str = parser.text

    # Post-processing - Clean up extra newlines as a result of HTML nesting, maintaining indents for readability
    plaintext_str = re.sub(NEWLINE_CLEANUP_PATTERN, "\n\n", plaintext_str)

    return plaintext_str


if __name__ == "__main__":
    json_str = """{
'strength': 'Medium',
'rationale': 'The announced progress in battery production for the Cybertruck suggests Tesla is advancing...',
'relation': "The monetization of technology is "supported" by Cybertruck Battery Production.",
'polarity': 'Positive',
'impact': 'High',
}"""

    try:
        s = json.loads(json_str)
        print(s)
    except json.JSONDecodeError as e:
        print(f"Failed to decode: {e}")

    print(json.loads(clean_to_json_if_needed(json_str)))
