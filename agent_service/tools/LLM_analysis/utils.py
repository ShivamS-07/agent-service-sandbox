import json
from typing import Any, Dict, List, Optional, Tuple

import Levenshtein

from agent_service.GPT.constants import GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_types.text import NewsText, TextCitation, TextGroup
from agent_service.tools.LLM_analysis.constants import (
    ANCHOR_HEADER,
    ANCHOR_REGEX,
    CITATION_HEADER,
    CITATION_SNIPPET_BUFFER_LEN,
    SENTENCE_REGEX,
    UNITS,
)
from agent_service.tools.LLM_analysis.prompts import KEY_PHRASE_PROMPT
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed


def strip_header(text_with_header: str) -> str:
    return text_with_header[text_with_header.find("\nText:\n") + 7 :]


def strip_units(phrase: str) -> str:
    for unit in UNITS:
        if phrase.endswith(unit):
            return phrase[: -len(unit)].strip()
    return phrase


def get_sentences(text: str) -> List[str]:
    paragraphs = text.replace("\n\n", "\n").split("\n")
    output_sentences = []
    for paragraph in paragraphs:
        output_sentences.extend(SENTENCE_REGEX.split(paragraph))
    return output_sentences


def find_best_sentence_match(snippet: str, sentences: List[str]) -> str:
    return min(
        [
            (Levenshtein.distance(snippet, sentence) / len(sentence), sentence)
            for sentence in sentences
            if len(sentence) > 3
        ]
    )[-1]


def get_initial_breakdown(GPT_ouput: str) -> Tuple[str, Optional[Dict[str, List[Dict[str, Any]]]]]:
    lines = GPT_ouput.replace("\n\n", "\n").split("\n")
    if (
        ANCHOR_HEADER in lines or CITATION_HEADER in lines
    ):  # GPT told not to do this, but it sometimes does anyway
        if ANCHOR_HEADER in lines:
            header_index = lines.index(ANCHOR_HEADER)
        elif CITATION_HEADER in lines:
            header_index = lines.index(CITATION_HEADER)
        main_text = "\n".join(lines[:header_index]).strip()
        citation_dict = "\n".join(lines[header_index + 1 :])
    else:
        main_text = "\n".join(lines[:-1])
        citation_dict = lines[-1]

    try:
        anchor_citation_dict = json.loads(clean_to_json_if_needed(citation_dict))
    except (json.JSONDecodeError, IndexError) as e:
        logger = get_prefect_logger(__name__)
        logger.warning(
            f"Got error `{e}` when loading `{citation_dict}` for citations, no citations included"
        )
        anchor_citation_dict = None
    return main_text, anchor_citation_dict


async def extract_citations_from_gpt_output(
    # Basic steps:
    # 1. Split up into main text and citations
    # 2. Iterate through the citation anchors in the text (with regex)
    # 3. Remove the anchor from the main text and get the index instead
    # 4. Get the citation snippet if any and check if it is the source text
    # 5. If the snippet is not in text, find a sentence which has key phrase, or otherwise similar
    # 6. Get the context around it, and package up the citation with text, index, and snippet
    GPT_ouput: str,
    text_group: TextGroup,
    context: PlanRunContext,
) -> Tuple[str, Optional[List[TextCitation]]]:
    logger = get_prefect_logger(__name__)
    llm = GPT(
        model=GPT4_O_MINI,
        context=create_gpt_context(GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID),
    )
    main_text, anchor_citation_dict = get_initial_breakdown(GPT_ouput)
    if anchor_citation_dict is None:
        return main_text, anchor_citation_dict
    final_text_bits = []
    final_citations = []
    index_counter = -1  # point to the last character in each added string
    last_end = 0
    for match in ANCHOR_REGEX.finditer(main_text):  # iterate over anchors
        anchor = match.group(1)
        punct = match.group(2)
        new_text = main_text[last_end : match.start()]  # text before anchor
        if punct:
            new_text += punct  # add punctuation after anchor
        final_text_bits.append(new_text)
        index_counter += len(
            new_text
        )  # advance the index so it points at the last character of text so far
        if anchor in anchor_citation_dict:
            citation_list = anchor_citation_dict[anchor]
            if not isinstance(citation_list, list):
                if isinstance(citation_list, dict):  # GPT probably forgot the list, just add it
                    citation_list = [citation_list]
                else:
                    logger.warning(
                        f"Skipping anchor due to problem with citation list: {anchor_citation_dict[anchor]}"
                    )
                    continue
            for citation_json in citation_list:
                if not isinstance(citation_json, dict) or "num" not in citation_json:
                    logger.warning(
                        f"Skipping citation due to problem with citation json: {citation_json}"
                    )
                    continue
                source_text_obj = text_group.convert_citation_num_to_text(citation_json["num"])
                if source_text_obj is None:
                    continue
                citation_snippet = citation_json.get("snippet", None)
                citation_snippet_context = None
                sentences = None
                if citation_snippet and not isinstance(source_text_obj, NewsText):
                    source_text_str = text_group.get_str_for_text(source_text_obj.id)
                    if source_text_str is not None:
                        source_text_str = strip_header(source_text_str)
                        idx = source_text_str.find(citation_snippet)
                        if idx == -1:  # GPT messed up, snippet is not a substring
                            if sentences is None:
                                sentences = get_sentences(source_text_str)
                            if len(sentences) == 0:  # No text, something is wrong, just skip
                                continue

                            # get a key phrase to help find the right sentence
                            key_phrase = strip_units(
                                await llm.do_chat_w_sys_prompt(
                                    Prompt(KEY_PHRASE_PROMPT, "KEY_PHRASE_PROMPT_STR").format(
                                        snippet=citation_snippet
                                    ),
                                    NO_PROMPT,
                                )
                            )
                            filtered_sentences = [
                                sentence for sentence in sentences if key_phrase in sentence
                            ]
                            if (
                                len(filtered_sentences) == 1
                            ):  # use the only sentence with key phrase
                                citation_snippet = filtered_sentences[0]
                            else:
                                if (
                                    len(filtered_sentences) == 0
                                ):  # bad key phrase too, use all sentences
                                    filtered_sentences = sentences
                                # find (key phrase) sentence with smallest string distance to snippet
                                citation_snippet = find_best_sentence_match(
                                    citation_snippet, filtered_sentences
                                )
                            idx = source_text_str.find(citation_snippet)  # Now definitely in text
                        citation_snippet_context = source_text_str[
                            max(0, idx - CITATION_SNIPPET_BUFFER_LEN) : idx
                            + len(citation_snippet)
                            + CITATION_SNIPPET_BUFFER_LEN
                        ]
                final_citations.append(
                    TextCitation(
                        source_text=source_text_obj,
                        citation_text_offset=index_counter,
                        citation_snippet=citation_snippet,
                        citation_snippet_context=citation_snippet_context,
                    )
                )
        else:
            logger.warning(f"anchor {anchor} in text did not have corresponding citations")

        last_end = match.end()
    if last_end != len(main_text):
        final_text_bits.append(main_text[last_end:])
    final_text = "".join(final_text_bits)
    return final_text, final_citations
