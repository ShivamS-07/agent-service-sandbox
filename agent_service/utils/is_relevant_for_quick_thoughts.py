from agent_service.GPT.constants import GPT4_O_MINI, Prompt
from agent_service.GPT.requests import GPT
from agent_service.types import Message
from agent_service.utils.logs import async_perf_logger

QUICK_THOUGHTS_RELEVANT_SYSTEM_PROMPT = Prompt(
    name="QUICK_THOUGHTS_RELEVANT_SYSTEM_PROMPT",
    template=(
        "You are given a message from a client, and need to decide "
        "if it can be answered by google search or not. "
        "In general if you think the answer for the client's ask can be found "
        "by a simple google search, then your response should be 'yes'. "
        "Your response must be a single word: 'yes' or 'no'. "
        "Here are some more guidelines to help you decide: "
        "\n- If client's ask is too long and complex with specified steps, "
        "it is not relevant for google search. "
        "\n- If want's any analysis related to their portfolio, "
        "it is not relevant for google search (e.g. 'how is my portfolio doing?', "
        "'write a commentary on my portoflio X'). "
        "\n- Any requests related to plotting graphs, or any other data visualization "
        "is not relevant for google search. "
        "\n- If the client is asking for stock prices, and market cap "
        "it is not relevant for google search since we have a tool for that. "
    ),
)
QUICK_THOUGHTS_RELEVANT_MAIN_PROMPT = Prompt(
    name="QUICK_THOUGHTS_RELEVANT_PROMPT",
    template=(
        "Your goal is to decide if the client's ask can be answered by a simple google search. "
        "\nHere is the client's ask: {message}"
        "\nSo can this message or question be answered by a simple google search? "
    ),
)


@async_perf_logger
async def is_relevant_for_quick_thoughts(message: Message) -> bool:

    llm = GPT(model=GPT4_O_MINI)
    res = await llm.do_chat_w_sys_prompt(
        main_prompt=QUICK_THOUGHTS_RELEVANT_MAIN_PROMPT.format(message=message.get_gpt_input()),
        sys_prompt=QUICK_THOUGHTS_RELEVANT_SYSTEM_PROMPT.format(),
    )
    return "yes" in res.lower()
