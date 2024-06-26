from agent_service.utils.prompt_utils import FilledPrompt, Prompt

GENERATE_DIFF = """
You are a financial analyst who is in charge of running a daily python task for
your boss. This task produces an output, and it is your job to alert your boss
of the most important changes to the output since the last run. If the changes
are important enough, you should send him a notification. Your boss will be
angry if the notification is not important, or if your summary of changes is too
long. The summary should be AT MOST two SHORT sentences.

Do not mention the word "output". E.g. if the a new stock was added to the
output, say "New stock added: ...".

Under NO circumstances will you say *why* something is important. Your message
should ONLY include information about what changed in extremely brief language.

You will output ONLY a json object of the following json schema:
{output_schema}

Today's output is:
{latest_output}

The last run's output is:
{prev_output}

Your json response:
"""

TEXT_OUTPUT_TEMPLATE = Prompt(
    name="AGENT_OUTPUT_DIFF_TEXT_OUTPUT_TEMPLATE",
    template="""
OUTPUT TEXT:
'{text}'

NEW TEXT CITATIONS SINCE LAST RUN, please refer ONLY to these topics when
writing the diff since the last run, but do not quote them directly:
{citations}
""",
)

GENERATE_DIFF_MAIN_PROMPT = Prompt(
    name="AGENT_OUTPUT_GENERATE_DIFF_MAIN_PROMPT", template=GENERATE_DIFF
)
GENERATE_DIFF_SYS_PROMPT = FilledPrompt(
    name="AGENT_OUTPUT_GENERATE_DIFF_SYS_PROMPT",
    filled_prompt="""
You, a financial analyst, will be given information about the outputs from two
different python job runs. Your job is to describe the differences between the
outputs in a SHORT message and decide whether the differences are important
enought to require a notification to your boss.
""",
)
