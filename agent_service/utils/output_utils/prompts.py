from agent_service.chatbot.prompts import AGENT_DESCRIPTION
from agent_service.utils.prompt_utils import FilledPrompt, Prompt

GENERATE_DIFF = """
You are a financial analyst who is in charge of running a daily python task for
your boss. This task produces an output, and it is your job to create a list of
the changes in the output since the last run. First, you will write a paragraph
which indicates all significant content changes. Modest rewording should never be
considered a significant content change, if there is nothing of any significance
you must simply output `No significant changes`.

If the changes are important enough, you should also send your boss a notification.
{notification_instructions}

You should be conservative, your boss will be very angry if you send a notification
for something that actually isn't important. However, you boss will also be angry if
some important change is missed.

Do not mention the word "output". NEVER mention integer stock ID's in your output!

If you choose to notify your boss, your output absolutely must contain some reference to the
instructions for notification from above and indicate explicitly in what way the changes
satisfy that criteria. Although it should be brief, your reference to the instructions
should be more than a simple repetition of those instructions. If you choose not to notify,
you should omit any justification.

Keep in mind that when you see a percentage, it is represented as a decimal. So
0.02 is equivalent to 2%. It is extremely important you take note of this. When
outputing text, make sure this is converted correctly to percentages.

You will output ONLY a json object of the following json schema:
{output_schema}

Today's output is:
---
{latest_output}
---

The last run's output is:
---
{prev_output}
---
{special_instructions}

Your json response:
"""

CUSTOM_NOTIFICATION_TEMPLATE = """
The boss has specifically asked for notifications in case of the following change(s):
{custom_notifications}
If the output differences you notice indicate (at least one of) the change(s) has occurred,
you should notify, otherwise, you should not.
"""

BASIC_NOTIFICATION_TEMPLATE = """
The boss wants notifications if there seems to have been some major event that has
greatly shifted the fortunes of major relevant stocks. You should be looking for
changes that are outside the normal day-to-day random variation (e.g. a sudden leap
or fall in a graph, a prominent stock jumping onto the top spot of a list after
not even being on it, etc.).
"""

NEW_TEXT_TEMPLATE = """
Below are the full list of texts used to create this document that were published between this run and
the last one. Your list of changes must only include information that are both different between the
two outputs AND which seem directly related to topics discussed in these new documents. Your boss is
only interested in the latest changes, not additions that actually are focused on old news.
---
{new_texts}
---
If the two outputs both have scores, and there is a noticeable change in the score (at least 0.1), you must
mention it and make sure your discussion is fully compatible with the change in score. For example, if the
change in score reflects an improvement in the market position of a stock, you MUST highlight any changes that
reflect that improvement.
"""

GENERATE_DIFF_MAIN_PROMPT = Prompt(
    name="AGENT_OUTPUT_GENERATE_DIFF_MAIN_PROMPT", template=GENERATE_DIFF
)
GENERATE_DIFF_SYS_PROMPT = FilledPrompt(
    name="AGENT_OUTPUT_GENERATE_DIFF_SYS_PROMPT",
    filled_prompt=f"""
{AGENT_DESCRIPTION}
You will be given information about the outputs from two different python job
runs. Your job is to describe the differences between the outputs in a SHORT
message and decide whether the differences are important enought to require a
notification to your boss. NEVER mention integer stock ID's in your output!
""",
)

SHORT_DIFF_SUMMARY_MAIN_PROMPT = Prompt(
    name="SHORT_DIFF_SUMMARY",
    template="""
You are an analyst who is doing daily updates of a report (which may include one or more sections)
You have already written a full list of the changes since your last report that includes all the
differences for all the sections (in bullet format), now you are going send a brief message to your
boss which summaries the most important findings. This should be no more than a sentence or two, your
boss is very busy and will be angry if you get wordy. You must not attempt to cover every section
if there are multiple sections, just pull out the most important information.
{custom_notifications}
Here is the full list of changes:
{diffs}
Now output your brief summary of the most important changes:
""",
)

SUMMARY_CUSTOM_NOTIFICATION_TEMPLATE = """
Your boss has left the following instructions about when he should be notified, you should focus
on these aspects whenever they apply:
{notification_criteria}
"""


DECIDE_NOTIFICATION_PROMPT_STR = """
You are a financial analyst who is in charge of running a daily python task for
your boss. This task produces an output, and it is your job to alert your boss
of the most important changes to the output since the last run. Here, you are simply
deciding whether or not to alert your boss.

{notification_instructions}

You should be conservative, your boss will be very angry if you send a notification
for something that actually isn't important. However, you boss will also be angry if
some important change is missed.

Output `Yes` if you decide to notify your boss, or `No` if not.

Today's output is:
{latest_output}

The last run's output is:
{prev_output}

Your response:
"""

DECIDE_NOTIFICATION_MAIN_PROMPT = Prompt(
    name="AGENT_OUTPUT_DECIDE_NOTIFICATION_MAIN_PROMPT", template=DECIDE_NOTIFICATION_PROMPT_STR
)
