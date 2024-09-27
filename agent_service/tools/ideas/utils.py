import random
from typing import List

from agent_service.GPT.constants import GPT4_O_MINI
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.text import Text, TextCitation
from agent_service.tools.ideas.constants import (
    MAX_TEXT_GROUP_TOKENS,
    MIN_TEXT_GROUP_TOKENS,
)


@io_type
class Idea(ComplexIOBase):
    title: str
    description: str
    citations: List[TextCitation]

    # TODO: Create functions that allow proper output
    # async def split_into_components(self) -> List[IOType]:
    #     pass

    # async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
    #     pass


async def create_small_text_groups(input_texts: List[Text]) -> List[List[Text]]:
    # this randomly splits up a potentially large group of texts into smaller groups of text which
    # are bounded to having a token length of no more than MAX_TEXT_GROUP_TOKENS + MIN_TEXT_GROUP_TOKENS
    tokenizer = GPTTokenizer(GPT4_O_MINI)
    random.shuffle(input_texts)
    output_text_groups = []
    curr_texts: List[Text] = []
    curr_token_count = 0
    text_strs: List[str] = await Text.get_all_strs(input_texts, include_header=True)  # type: ignore
    for text_obj, text_str in zip(input_texts, text_strs):
        text_token_count = tokenizer.get_token_length(text_str)
        if text_token_count + curr_token_count > MAX_TEXT_GROUP_TOKENS:
            output_text_groups.append(curr_texts)
            curr_texts = [text_obj]
            curr_token_count = text_token_count
        else:
            curr_texts.append(text_obj)
            curr_token_count += text_token_count

    # get rid of the last group if it is too small, better to have a slightly oversized group then
    # a super small one
    if len(output_text_groups) > 1 and curr_token_count < MIN_TEXT_GROUP_TOKENS:
        output_text_groups[-2].extend(output_text_groups[-1])
        del output_text_groups[-1]

    return output_text_groups
