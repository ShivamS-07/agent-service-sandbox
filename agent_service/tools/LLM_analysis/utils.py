import copy
import json
from typing import Any, Dict, List, Optional, Tuple

import Levenshtein

from agent_service.GPT.constants import GPT4_O, GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import ComplexIOBase, IOTypeBase
from agent_service.io_types.text import (
    DEFAULT_TEXT_TYPE,
    NewsText,
    StockText,
    Text,
    TextCitation,
    TextCitationGroup,
    TextGroup,
)
from agent_service.tools.LLM_analysis.constants import (
    ANCHOR_HEADER,
    ANCHOR_REGEX,
    CITATION_HEADER,
    CITATION_SNIPPET_BUFFER_LEN,
    SENTENCE_REGEX,
    UNITS,
)
from agent_service.tools.LLM_analysis.prompts import (
    KEY_PHRASE_PROMPT,
    SECOND_ORDER_CITATION_PROMPT,
    TEXT_SNIPPET_RELEVANCY_MAIN_PROMPT,
    TEXT_SNIPPET_RELEVANCY_SYS_PROMPT,
)
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.output_utils.output_construction import PreparedOutput
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
    # remove very short sentences since they won't match properly
    output_sentences = [sentence for sentence in output_sentences if len(sentence) > 3]
    return output_sentences


def find_best_sentence_match(snippet: str, sentences: List[str]) -> str:
    return min(
        [
            (Levenshtein.distance(snippet, sentence) / len(sentence), sentence)
            for sentence in sentences
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
    final_citations = None
    index_counter = -1  # point to the last character in each added string
    last_end = 0
    for match in ANCHOR_REGEX.finditer(main_text):  # iterate over anchors
        if (
            final_citations is None
        ):  # at least one citation, so not a formatting fail/pure hallucination
            final_citations = []
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
                if source_text_obj is None or source_text_obj.text_type == DEFAULT_TEXT_TYPE:
                    # Do not create citations for text we generate, do second order citations
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


def get_original_cite_count(citations: List[TextCitation]) -> int:
    return len(
        [citation for citation in citations if citation.source_text.text_type != DEFAULT_TEXT_TYPE]
    )


def get_all_text_citations(obj: IOTypeBase) -> List[TextCitation]:
    if isinstance(obj, List):
        return [citation for sub_obj in obj for citation in get_all_text_citations(sub_obj)]
    elif isinstance(obj, PreparedOutput):
        return get_all_text_citations(obj.val)
    elif isinstance(obj, ComplexIOBase):
        return [
            citation for citation in obj.get_all_citations() if isinstance(citation, TextCitation)
        ]
    else:
        return []


async def get_second_order_citations(
    main_text: str, old_citations: List[TextCitation], context: PlanRunContext
) -> List[TextCitation]:
    if not old_citations:
        return []

    citation_group = TextCitationGroup(val=old_citations)
    citation_str = await citation_group.convert_to_str()

    sentences = [sentence for sentence in get_sentences(main_text) if len(sentence) > 20]

    sentences_str = "\n".join(f"{i}. {sentence}" for i, sentence in enumerate(sentences, start=1))

    llm = GPT(
        model=GPT4_O,
        context=create_gpt_context(GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID),
    )

    main_prompt = SECOND_ORDER_CITATION_PROMPT.format(sents=sentences_str, snippets=citation_str)

    result = await llm.do_chat_w_sys_prompt(main_prompt, NO_PROMPT, output_json=True)

    # json mode can't output integer keys, so need to change to ints manually
    citation_mapping = {
        int(key): value for key, value in json.loads(clean_to_json_if_needed(result)).items()
    }

    new_citations = []

    for num, sentence in enumerate(sentences, start=1):
        if num in citation_mapping and citation_mapping[num] and sentence in main_text:
            sentence_offset = main_text.index(sentence) + len(sentence) - 1
            for citation_num in citation_mapping[num]:
                citation = citation_group.convert_citation_num_to_citation(citation_num)
                if citation:
                    new_citation = copy.deepcopy(citation)
                    new_citation.citation_text_offset = sentence_offset
                    new_citations.append(new_citation)

    return new_citations


async def classify_stock_text_relevancy_for_profile(
    text: str,
    profiles_str: str,
    company_name: str,
    llm: GPT,
) -> bool:

    chopped_text_str = GPTTokenizer(model=llm.model).do_truncation_if_needed(
        truncate_str=text,
        other_prompt_strs=[
            TEXT_SNIPPET_RELEVANCY_MAIN_PROMPT.template,
            TEXT_SNIPPET_RELEVANCY_SYS_PROMPT.template,
            company_name,
            profiles_str,
        ],
    )

    output = await llm.do_chat_w_sys_prompt(
        main_prompt=TEXT_SNIPPET_RELEVANCY_MAIN_PROMPT.format(
            company_name=company_name,
            profiles=profiles_str,
            text_snippet=chopped_text_str,
        ),
        sys_prompt=TEXT_SNIPPET_RELEVANCY_SYS_PROMPT.format(),
        max_tokens=500,
    )

    try:
        decision = int(output.strip()[0])
        if decision == 1:
            return True
        else:
            return False
    except (json.JSONDecodeError, ValueError, IndexError):
        logger = get_prefect_logger(__name__)
        logger.warning(
            f"Failed to get text snippet relevancy output, got '{output}'", exc_info=True
        )
        return False


async def classify_stock_text_relevancies_for_profile(
    texts: List[StockText], profiles_str: str, context: PlanRunContext
) -> List[StockText]:
    filtered_texts: List[StockText] = []
    text_strs = await Text.get_all_strs(texts, include_header=True, include_timestamps=False)

    llm = GPT(
        model=GPT4_O_MINI,
        context=create_gpt_context(GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID),
    )

    tasks = []
    for i, text in enumerate(texts):
        if isinstance(text, StockText):
            text_str = text_strs[i]
            # If StockText, there must be a stock_id
            company_name = text.stock_id.company_name  # type: ignore
            tasks.append(
                classify_stock_text_relevancy_for_profile(
                    text=text_str, profiles_str=profiles_str, company_name=company_name, llm=llm
                )
            )

    results = await gather_with_concurrency(tasks, n=200)
    for i, relevancy_descision in enumerate(results):
        if relevancy_descision:
            filtered_texts.append(texts[i])
    return filtered_texts
