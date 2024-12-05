from typing import List, Optional, Set

from agent_service.GPT.constants import GPT4_O_MINI
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_types.idea import Idea
from agent_service.io_types.text import StockText, Text, TextCitation
from agent_service.tools.ideas.constants import (
    MAX_TEXT_GROUP_TOKENS,
    MIN_TEXT_GROUP_TOKENS,
)
from agent_service.types import AgentUserSettings
from agent_service.utils.feature_flags import get_ld_flag


async def create_small_text_groups(input_texts: List[Text]) -> List[List[Text]]:
    # this splits up a potentially large group of texts into smaller groups of text which
    # are bounded to having a token length of no more than MAX_TEXT_GROUP_TOKENS + MIN_TEXT_GROUP_TOKENS
    tokenizer = GPTTokenizer(GPT4_O_MINI)
    # TODO: more testing whether randomness is good or not, maybe a better solution with embeddings
    # random.shuffle(input_texts)
    output_text_groups = []
    curr_texts: List[Text] = []
    curr_token_count = 0
    text_strs: List[str] = await Text.get_all_strs(  # type: ignore
        input_texts, include_header=True, include_symbols=True
    )
    for text_obj, text_str in zip(input_texts, text_strs):
        text_token_count = tokenizer.get_token_length(text_str)
        if text_token_count + curr_token_count > MAX_TEXT_GROUP_TOKENS:
            output_text_groups.append(curr_texts)
            curr_texts = [text_obj]
            curr_token_count = text_token_count
        else:
            curr_texts.append(text_obj)
            curr_token_count += text_token_count

    output_text_groups.append(curr_texts)
    # get rid of the last group if it is too small, better to have a slightly oversized group then
    # a super small one
    if len(output_text_groups) > 1 and curr_token_count < MIN_TEXT_GROUP_TOKENS:
        output_text_groups[-2].extend(output_text_groups[-1])
        del output_text_groups[-1]

    return output_text_groups


def ideas_enabled(user_id: Optional[str], user_settings: Optional[AgentUserSettings]) -> bool:
    result = get_ld_flag("agent-svc-ideas-tools-enabled", default=False, user_context=user_id)
    return result


def distinct_stock_count(source_texts: List[Text]) -> int:
    stocks = set()
    for text in source_texts:
        if isinstance(text, StockText) and text.stock_id:
            stocks.add(text.stock_id)
    return len(stocks)


def distinct_text_count(source_texts: List[Text]) -> int:
    text_ids = set()
    for text in source_texts:
        text_ids.add(text.get_original_text_id())
    return len(text_ids)


def get_source_texts(idea: Idea) -> Set[Text]:
    old_source_texts = [
        citation.source_text
        for citation in idea.description.history[-1].citations
        if isinstance(citation, TextCitation)
    ]
    for old_source_text in old_source_texts:
        old_source_text.reset_id()

    return set(old_source_texts)
