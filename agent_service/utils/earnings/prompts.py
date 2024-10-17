# flake8: noqa

from agent_service.utils.prompt_utils import Prompt

TRANSCRIPT_PARTITION_MAIN_PROMPT_STR = """Each line of the transcript will be labeled by a number, output each partition on a new line, denote each partition by the following format N,M where N is the starting line number and M is the ending line number, do not output anything else in addition. Each line number must belong in a partition. If you feel that one line should be partitioned as its own section then simply output N,N for that line. The starting line number for the next partition should use the ending line number of the previous partition incremented by one. Your output should look something like this:
1,5
6,9
11,19
20,20
...
The numbers in the above output are just examples, this is just to show you the structure of the output. Do not include a line number that does not exist. Below is the transcript:
{transcript_text}"""

TRANSCRIPT_PARTITION_SYS_PROMPT_STR = """You are a junior financial analyst preparing data for your senior analyst. You have been given a full earning transcript of a recent company's earnings call. However, the full transcript is too large. Your task is to break the transcript down into more manageable parts while ensuring the pieces you break down are self-contained, this is to say that the snippets you partition are not cut off while discussing a certain topic, and questions with their respective answers are bundled together. Some lines may be self contained, for example a very long line talking about a self-contained topic, the introduction line, or the last line of the transcript signalling the end of the call."""

TRANSCRIPT_PARTITION_SYS_PROMPT = Prompt(
    name="TRANSCRIPT_PARTITION_SYS_PROMPT",
    template=TRANSCRIPT_PARTITION_SYS_PROMPT_STR,
)
TRANSCRIPT_PARTITION_MAIN_PROMPT = Prompt(
    name="TRANSCRIPT_PARTITION_MAIN_PROMPT",
    template=TRANSCRIPT_PARTITION_MAIN_PROMPT_STR,
)
