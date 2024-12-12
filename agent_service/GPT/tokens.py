from typing import List

import tiktoken

from agent_service.GPT.constants import DEFAULT_OUTPUT_LEN, GPT4_O, MAX_TOKENS


class GPTTokenizer:
    def __init__(self, model: str) -> None:
        self.model = model
        if model.startswith("claude"):
            # We will use gpt-4o encoder since anthropic doesn't yet provide us a tokenizer
            self.encoder = tiktoken.encoding_for_model(GPT4_O)
        else:
            self.encoder = tiktoken.encoding_for_model(model)

    def do_truncation_if_needed(
        self, truncate_str: str, other_prompt_strs: List[str], output_len: int = DEFAULT_OUTPUT_LEN
    ) -> str:
        used = self.get_token_length("\n".join(other_prompt_strs))
        return self.chop_input_to_allowed_length(truncate_str, used, output_len)

    def do_multi_truncation_if_needed(
        self,
        flex_strs: List[str],
        other_prompt_strs: List[str],
        output_len: int = DEFAULT_OUTPUT_LEN,
    ) -> List[str]:
        used = self.get_token_length("\n".join(other_prompt_strs))
        tokens_per_flex_str = [self.get_token_length(flex_str) for flex_str in flex_strs]
        flex_tokens_sum = sum(tokens_per_flex_str)
        if flex_tokens_sum + output_len + used > MAX_TOKENS[self.model]:
            available_tokens = MAX_TOKENS[self.model] - (output_len + used)
            allowed_tokens_per_str = [
                int(available_tokens * (flex_tokens / flex_tokens_sum))
                for flex_tokens in tokens_per_flex_str
            ]
            output = []
            for flex_str, flex_allowed_len in zip(flex_strs, allowed_tokens_per_str):
                flex_input_tokens = self.encoder.encode(flex_str)
                clipped_flex_input = self.encoder.decode(flex_input_tokens[:flex_allowed_len])
                output.append(clipped_flex_input)
            return output
        else:
            return flex_strs

    def chop_input_to_allowed_length(
        self,
        flexible_input: str,
        fixed_input_len: int,
        output_len: int = DEFAULT_OUTPUT_LEN,
    ) -> str:
        """
        This function uses the tiktoken tokenizer to ensure to chop the input as needed to
        ensure as much input can be used as possible without going over the GPT model limits

            flexible_input (str) : This is input that can be chopped to length if needed
                                e.g. text from 10-k
            fixed_input_len (int): number of tokens in non-flexible input
            model (str): The GPT model, different models have different maximum lengths
            output_len (int): number of tokens which should be reserved for the output

            Returns flexible input chopped so everything else will fit

        """
        used_tokens = output_len + fixed_input_len
        max_tokens = MAX_TOKENS[self.model]
        if used_tokens >= max_tokens:
            raise ValueError("Required tokens already at limit for GPT request")
        flex_input_tokens = self.encoder.encode(flexible_input)
        clipped_flex_input = self.encoder.decode(flex_input_tokens[: max_tokens - used_tokens])
        return clipped_flex_input

    def get_token_length(self, input: str) -> int:
        # Get the number of tokens in the provided input string
        return len(self.encoder.encode(input))
