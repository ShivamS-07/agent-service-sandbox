from typing import List

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import IOType, dump_io_type
from agent_service.io_types.text import Text
from agent_service.io_types.text_objects import StockTextObject
from agent_service.tool import ToolArgs, ToolCategory, ToolRegistry, tool
from agent_service.types import PlanRunContext
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.output_utils.output_construction import PreparedOutput
from agent_service.utils.postgres import SyncBoostedPG
from agent_service.utils.prompt_utils import Prompt

ANALYZE_OUTPUT_SYS_PROMPT_STR = "You are an financial analyst supporting a client request that requires interactive data analysis. In particular, you have already created one or more outputs for the client, and now need to do some further analysis of those outputs according to the client's request. The outputs can be of various forms, including a table of data (in column format), a graph of data, a text (of various kinds), a list of texts, and others. You will be provided with a specific purpose you are trying to satisfy and the entire chat context with the client to help you contextualize the request and pull in exactly the data you need to prepare your own output. Be aware that the output is ultimately a Python data object and as such may contain things like json keys and uuids which must NEVER be shown directly to the client. The client is a finance expert, but they are not a programmer and all technical details must be hidden from them. If there is a table, please be careful pulling rows out information, you will often be looking for a particular stock, you will need to count the number of steps into the column to find the stock, and then take the same number of steps into the other columns you need to get the to the specific value you are looking for. Graphs, on the other hand, will often consist basically of lists of lists of datapoints, one list of datapoints for each stock (or similar). If the client is asking you to describe a graph, they are referring to graph they are seeing after the points you have are plotted, for instance if the datapoints are unchanging over time then the graph is flat, or a rapid increase could be described a sharp uptick. Note that sometimes the user may ask a question or make a request that involves the output data but where the information required is not actually contained with it. For example, the user may ask why a particular uptick in the graph happened, but there is only a graph in the output, with no information about why it happened. When the user asks such a question, you should both a) try to demonstrate as much general knowledge of the situation as you can (for example identifying the exact date the uptick occurred), and perhaps provide other information that you have which is relevant (e.g. this stock is highly volatile) b) admit ignorance is so far as the specific information that the user requests is not actually in the previous outputs, for instance you note it would require reading a document the text of which is not directly available to you without rerunning your work. In such circumstances, you can suggest that the user try adding a follow up request specifically targeted at an analysis of those documents or documents, or which pull in other information that is not currently visible to the user. Whatever you do, it is critical that you never pretend to have knowledge you do not actually have, it will be easily found out and you will be fired for lying to the client."  # noqa: E501

ANALYZE_OUTPUT_MAIN_PROMPT_STR = "Based on a client request or question, pull information from previous outputs and output a text which satisfies the client need. Here is the general purpose of the question/requestion: {purpose}. Here is the full chat transcript with the client, delimited by ---:\n---\n{chat_context}\n---\n. Here are the relevant previous output objects (possibly only one), the set of previous outputs is delimited by --- and if there is more than one each output is delimited by ***:\n---{outputs}\n---\nNow write your response that satisfies the client need:\n"  # noqa: E501

ANALYZE_OUTPUT_SYS_PROMPT = Prompt(ANALYZE_OUTPUT_SYS_PROMPT_STR, "ANALYZE_OUTPUT_SYS_PROMPT")
ANALYZE_OUTPUT_MAIN_PROMPT = Prompt(ANALYZE_OUTPUT_MAIN_PROMPT_STR, "ANALYZE_OUTPUT_MAIN_PROMPT")


class OutputArgs(ToolArgs):
    object_to_output: IOType
    title: str


@tool(
    description="""This is a very important function that displays any object
to the user, along with a short title for the object. EVERY plan you come up with
should have at least one call to this function. Objects will be displayed in the
same order the function is called in. For example, assuming there are three
objects stored in variables called "text", "graph", and "table", the calls:

   result1 = output(object_to_output=text, title="Description of Stocks")
   result2 = output(object_to_output=graph, title="Price Graph")
   result3 = output(object_to_output=table, title="Table of Weights")

Will show the objects top to bottom in that order. Please display only what the
user asks for, and no other extraneous information. Please only display every
variable once, otherwise the user will see duplicates. Remember you need to
assign this function call to a variable!!
""",
    category=ToolCategory.OUTPUT,
    is_visible=False,
    is_output_tool=True,
    store_output=False,
)
async def prepare_output(args: OutputArgs, context: PlanRunContext) -> PreparedOutput:
    if isinstance(args.object_to_output, Text) and context.stock_info:
        # Handle text object resolution.
        text_val = (await args.object_to_output.get()).val
        stock_text_objects = await StockTextObject.find_stock_references_in_text(
            text=text_val, stocks=list(context.stock_info)
        )
        args.object_to_output.text_objects.extend(stock_text_objects)
    return PreparedOutput(title=args.title, val=args.object_to_output)


class AnalyzeOutputInputs(ToolArgs):
    outputs: List[PreparedOutput]
    purpose: str


@tool(
    description=(
        "This function allows for inspection of any outputs with have been produced earlier in "
        "the plan, creating text whose content is aimed at satisfying some related purpose indicated by the client. "
        "Specifically, There are two major use cases for this tool:\n"
        "1. When the client asks specifically for a text output which includes information that is available "
        "only via an existing list of stocks and/or a table, this is the way to read that data, and then either  "
        "present it to the user as text or integrate the data into a larger summary. Usually in this case you will "
        "pass a list of one output, assuming the output the user wants is clear and doesn't require synthesis of "
        "multiple outputs; if they want a summary of information from multiple outputs, include everything relevant "
        "in the list of outputs\n"
        "2. After an initial plan run is complete, if a user asks a follow up question which seems to refer to "
        "an output you have shown to the user via the prepare_output tool, but there is insufficient context to "
        "be sure which specific output, you may decide to simply inspect all previous outputs for an answer to "
        "the question. "
        "An simple example of when you'd use this tool: `give me a line graph of Apple performance over the last month "
        "and then write a summary of the major changes.` In this case, you would use this tool to write the summary,"
        "not the summary tool, because that tool only accepts text input, not table input."
        "You should also use this tool if the user asks follow-up questions that specifically ask about "
        "something you've already generated, or asks a question mentioning `you`, e.g. `what you mean by X?`"
        "or something else which requires context you don't seem to have in order to to interpret, e.g. "
        "`what does X refer to here here?` where `X` hasn't been mentioned anywhere else in your conversation."
        "Note that this tool ONLY inspects the output. If you need to combine inspection of an output with "
        "additional data analysis, you should first inspect the output(s) to get the information you need from "
        "it, and then pass the resulting text with other relevant texts to the summarize tool. You should "
        "generally avoid calling this tool in cases where previous output is very unlikely to contain exactly the "
        "information the user requires. "
        "Note that occasionally the outputs might be cutoff when there is too much data, if so, just work with "
        "what you have, do not tell the user anything about this issue in your output."
        "Generally you should never call this tool on text output alone, it is unnecessary. If further analysis "
        "of text is required after some follow-up question, use one of the LLM analysis tools and output the result."
    ),
    category=ToolCategory.OUTPUT,
)
async def analyze_outputs(args: AnalyzeOutputInputs, context: PlanRunContext) -> Text:

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(model=GPT4_O, context=gpt_context)
    pg = SyncBoostedPG()
    if context.chat:
        chat_str = context.chat.get_gpt_input()
    else:
        chat_str = ""
    outputs_strs = [dump_io_type(await output.to_rich_output(pg=pg)) for output in args.outputs]
    outputs_strs = GPTTokenizer(GPT4_O).do_multi_truncation_if_needed(
        outputs_strs,
        [
            ANALYZE_OUTPUT_MAIN_PROMPT.template,
            ANALYZE_OUTPUT_SYS_PROMPT.template,
            chat_str,
            args.purpose,
        ],
    )
    outputs_str = "***".join(outputs_strs)
    main_prompt = ANALYZE_OUTPUT_MAIN_PROMPT.format(
        purpose=args.purpose, chat_context=chat_str, outputs=outputs_str
    )
    result = await llm.do_chat_w_sys_prompt(main_prompt, ANALYZE_OUTPUT_SYS_PROMPT.format())
    return Text(val=result)


# Hack for backwards compatibility
ToolRegistry._REGISTRY_CATEGORY_MAP[ToolCategory.OUTPUT]["output"] = (
    ToolRegistry._REGISTRY_CATEGORY_MAP[ToolCategory.OUTPUT]["prepare_output"]
)
ToolRegistry._REGISTRY_ALL_TOOLS_MAP["output"] = ToolRegistry._REGISTRY_CATEGORY_MAP[
    ToolCategory.OUTPUT
]["prepare_output"]
