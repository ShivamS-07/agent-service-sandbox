import json
from typing import List, Tuple

from agent_service.GPT.constants import GPT4_O, GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import HistoryEntry
from agent_service.io_types.idea import Idea
from agent_service.io_types.text import Text, TextGroup
from agent_service.planner.errors import EmptyInputError
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.ideas.constants import (
    DEFAULT_MAX_IDEAS,
    MAX_BRAINSTORM_TRIES,
    MIN_IDEAS,
)
from agent_service.tools.ideas.prompts import (
    BRAINSTORM_IDEAS_DESCRIPTION,
    FINAL_IDEA_MAIN_PROMPT,
    FINAL_IDEA_SYS_PROMPT,
    IDEA_CLUSTER_MAIN_PROMPT_STR,
    IDEA_CLUSTER_SYS_PROMPT_STR,
    INITIAL_BRAINSTORM_PROMPT,
)
from agent_service.tools.ideas.utils import create_small_text_groups
from agent_service.tools.LLM_analysis.constants import MAX_CITATION_TRIES
from agent_service.tools.LLM_analysis.utils import extract_citations_from_gpt_output
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.smart_clustering import SmartClustering
from agent_service.utils.string_utils import clean_to_json_if_needed


async def initial_brainstorm(
    texts: List[Text], idea_definition: str, context: PlanRunContext
) -> List[Tuple[str, List[Text], int]]:
    logger = get_prefect_logger(__name__)
    text_group = TextGroup(val=texts)
    text_str = await Text.get_all_strs(text_group, include_header=True, text_group_numbering=True)
    if context.chat:
        chat_str = context.chat.get_gpt_input()
    else:
        chat_str = ""

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    cheap_llm = GPT(context=gpt_context, model=GPT4_O_MINI)

    main_prompt = INITIAL_BRAINSTORM_PROMPT.format(
        idea_definition=idea_definition, chat_context=chat_str, texts=text_str
    )

    result = await cheap_llm.do_chat_w_sys_prompt(main_prompt, NO_PROMPT, output_json=True)

    final_output: List[Tuple[str, List[Text], int]] = []
    tries = 0
    success = False
    while not success and tries < MAX_BRAINSTORM_TRIES:
        try:
            json_ideas_output = json.loads(clean_to_json_if_needed(result))
            final_output = []
            for rank, idea_dict in enumerate(json_ideas_output, start=1):
                sources = [
                    text_group.convert_citation_num_to_text(source)
                    for source in idea_dict["sources"]
                ]
                sources = [text for text in sources if text]
                if sources:
                    final_output.append((idea_dict["idea"], sources, rank))  # type: ignore
            success = True
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Failed to parse correct json from initial brainstorm output: {result}")
            result = await cheap_llm.do_chat_w_sys_prompt(
                main_prompt, NO_PROMPT, output_json=True, no_cache=True
            )
            tries += 1

    return final_output


async def cluster_ideas(
    idea_definition: str, initial_ideas: List[Tuple[str, List[Text], int]]
) -> List[Tuple[List[str], List[Text], float]]:
    clusterer = SmartClustering(
        identifier="IDEA_BRAINSTORM",
        items=idea_definition,
        sys_prompt=IDEA_CLUSTER_SYS_PROMPT_STR,
        main_prompt=IDEA_CLUSTER_MAIN_PROMPT_STR,
    )
    idea_strs = [idea for idea, _, _ in initial_ideas]
    groups = await clusterer.apply_smart_clustering(idea_strs)
    # create singleton groups for unclustered ideas
    not_in_group = set(range(len(idea_strs))) - set([idx for group in groups for idx in group])
    for idx in not_in_group:
        groups.append([idx])
    output_idea_clusters = []
    for group in groups:
        idea_strs = []
        all_idea_citations = set()
        inverse_rank_sum = 0.0
        for idx in group:
            idea_str, idea_citations, idea_rank = initial_ideas[idx]
            idea_strs.append(idea_str)
            all_idea_citations.update(idea_citations)
            inverse_rank_sum += 1 / idea_rank
        output_idea_clusters.append((idea_strs, list(all_idea_citations), inverse_rank_sum))
    return output_idea_clusters


async def create_final_idea(
    idea_definition: str,
    idea_formulations: List[str],
    idea_relevant_texts: List[Text],
    context: PlanRunContext,
) -> Idea:
    logger = get_prefect_logger(__name__)
    if context.chat:
        chat_str = context.chat.get_gpt_input()
    else:
        chat_str = ""

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)

    text_group = TextGroup(val=idea_relevant_texts)
    text_str: str = await Text.get_all_strs(  # type: ignore
        text_group, include_header=True, text_group_numbering=True
    )
    ideas_str = "\n".join(idea_formulations)

    text_str = GPTTokenizer(GPT4_O).do_truncation_if_needed(
        text_str,
        [
            FINAL_IDEA_MAIN_PROMPT.template,
            FINAL_IDEA_SYS_PROMPT.template,
            chat_str,
            ideas_str,
            idea_definition,
        ],
    )

    main_prompt = FINAL_IDEA_MAIN_PROMPT.format(
        idea_definition=idea_definition,
        chat_context=chat_str,
        initial_ideas=ideas_str,
        source_documents=text_str,
    )
    sys_prompt = FINAL_IDEA_SYS_PROMPT.format()

    result = await llm.do_chat_w_sys_prompt(main_prompt, sys_prompt)

    lines = result.strip().split("\n")
    title = lines[0]
    initial_text = "\n".join(lines[1:])
    final_text, citations = await extract_citations_from_gpt_output(
        initial_text, text_group, context
    )
    tries = 0
    while citations is None and tries < MAX_CITATION_TRIES:  # failed to load citations, retry
        logger.warning(f"Retrying after no citations after {result}")
        result = await llm.do_chat_w_sys_prompt(
            main_prompt, sys_prompt, no_cache=True, temperature=0.1 * (tries + 1)
        )
        lines = result.strip().split("\n")
        title = lines[0]
        initial_text = "\n".join(lines[1:])
        final_text, citations = await extract_citations_from_gpt_output(
            initial_text, text_group, context
        )
        tries += 1

    if citations is None:
        citations = []

    description: Text = Text(val=final_text)
    description = description.inject_history_entry(HistoryEntry(citations=citations))  # type: ignore

    return Idea(title=title, description=description)


class BrainstormIdeasFromTextsInput(ToolArgs):
    idea_definition: str
    texts: List[Text]
    max_ideas: int = DEFAULT_MAX_IDEAS


@tool(description=BRAINSTORM_IDEAS_DESCRIPTION, category=ToolCategory.IDEAS, enabled=False)
async def brainstorm_ideas_from_text(
    args: BrainstormIdeasFromTextsInput, context: PlanRunContext
) -> List[Idea]:
    if len(args.texts) == 0:
        raise EmptyInputError("Missing input texts to idea brainstorm")
    text_groups = await create_small_text_groups(args.texts)
    tasks = []
    # do brainstorming for each group that can fit in a single GPT context call
    for text_group in text_groups:
        tasks.append(initial_brainstorm(text_group, args.idea_definition, context))
    result = await gather_with_concurrency(tasks)
    initial_ideas = [idea for idea_groups in result for idea in idea_groups]
    # cluster ideas across the runs
    clustered_ideas = await cluster_ideas(args.idea_definition, initial_ideas)
    # only include ideas with at least two citations provided there are at least 5 ideas with 2 citations
    two_citation_clustered_ideas = [idea for idea in clustered_ideas if len(idea[1]) > 1]
    if len(two_citation_clustered_ideas) >= MIN_IDEAS:
        clustered_ideas = two_citation_clustered_ideas
    # sort ideas by sum of num citations and sum of inverse rank and chop to max
    clustered_ideas.sort(key=lambda x: -(len(x[1]) + x[2]))
    clustered_ideas = clustered_ideas[: args.max_ideas]
    # generate final ideas with references
    final_tasks = [
        create_final_idea(args.idea_definition, idea_formulations, idea_relevant_texts, context)
        for idea_formulations, idea_relevant_texts, _ in clustered_ideas
    ]
    final_ideas = await gather_with_concurrency(final_tasks, n=len(clustered_ideas))

    return final_ideas
