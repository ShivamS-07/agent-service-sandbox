import tiktoken

from agent_service.GPT.constants import DEFAULT_OUTPUT_LEN, MAX_TOKENS


class GPTTokenizer:
    def __init__(self, model: str) -> None:
        self.model = model
        self.encoder = tiktoken.encoding_for_model(model)

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
