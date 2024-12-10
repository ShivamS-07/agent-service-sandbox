from llm_client.datamodels import LLMFunction
from llm_client.functions.llm_func import LLMFuncArgs

from agent_service.external.gemini_client import GeminiClient
from agent_service.GPT.constants import GPT4_O_MINI
from agent_service.GPT.requests import GPT
from agent_service.q_and_a.utils import QAContext
from agent_service.utils.async_db import get_async_db
from agent_service.utils.logs import async_perf_logger
from agent_service.utils.prompt_utils import FilledPrompt, Prompt

Q_AND_A_SUMMARIZE_GOOGLE_GROUNDING_PROMPT = Prompt(
    name="Q_AND_A_SUMMARIZE_GOOGLE_GROUNDING_PROMPT",
    template="""
This is the result from a google search you just made, please condense the info
into smaller piece of text. Ensure your reply is clear, relevant, and less than
100 words.

Result to condense:
{text}
""",
)


class GeneralQuestionArgs(LLMFuncArgs):
    question: str


@async_perf_logger
async def ask_general_question(
    args: GeneralQuestionArgs,
    context: QAContext,
) -> str:
    db = get_async_db()
    gemini_client = GeminiClient(context=context.gpt_context)
    summary_llm = GPT(model=GPT4_O_MINI, context=context.gpt_context)
    query = f"""
    Please look for info on this question: {args.question}

    The chat context of the question is:
    {context.chat_context.get_gpt_input()}
    """

    result = await gemini_client.query_google_grounding(query=query, db=db.pg)
    summarized = await summary_llm.do_chat_w_sys_prompt(
        main_prompt=Q_AND_A_SUMMARIZE_GOOGLE_GROUNDING_PROMPT.format(text=result),
        sys_prompt=FilledPrompt(filled_prompt=""),
    )
    return summarized


ASK_GENERAL_QUESTION_FUNC = LLMFunction(
    name="ask_general_question",
    args=GeneralQuestionArgs,
    func=ask_general_question,
    description="""
Anything unrelated to the specific workflow that the user is looking at. Only
use this if the user asks some information that you would need to search the
internet to find out. If you can already answer the question, no need to call
this.
    """,
)
