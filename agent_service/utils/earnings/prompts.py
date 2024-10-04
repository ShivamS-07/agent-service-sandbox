# flake8: noqa

from agent_service.utils.prompt_utils import Prompt

TRANSCRIPT_PARTITION_MAIN_PROMPT_STR = """Each line of the transcript will be labeled by a number, output each partition on a new line, denote each partition by the following format N,M where N is the starting line number and M is the ending line number. Each line must belong in a partition. The starting line number for the next partition should use the ending line number of the previous partition incremented by one. Below is the transcript:
{transcript_text}"""

TRANSCRIPT_PARTITION_SYS_PROMPT_STR = """You are a junior financial analyst preparing data for your senior analyst. You have been given a full earning transcript of a recent company's earnings call. However, the full transcript is too large. Your task is to break the transcript down into more manageable parts while ensuring the pieces you break down are self-contained, this is to say that the snippets you partition are not cut off while discussing a certain topic, and questions with their respective answers are bundled together. """

TRANSCRIPT_PARTITION_SYS_PROMPT = Prompt(
    name="TRANSCRIPT_PARTITION_SYS_PROMPT",
    template=TRANSCRIPT_PARTITION_SYS_PROMPT_STR,
)
TRANSCRIPT_PARTITION_MAIN_PROMPT = Prompt(
    name="TRANSCRIPT_PARTITION_MAIN_PROMPT",
    template=TRANSCRIPT_PARTITION_MAIN_PROMPT_STR,
)
