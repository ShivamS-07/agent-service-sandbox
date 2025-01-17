import inspect
import json
from typing import Coroutine, Dict, List, Optional, Set, Tuple

from agent_service.GPT.constants import GPT4_O, GPT4_O_MINI, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import HistoryEntry
from agent_service.io_types.idea import Idea
from agent_service.io_types.text import Text, TextGroup
from agent_service.planner.errors import EmptyInputError, EmptyOutputError
from agent_service.tool import ToolArgs, ToolCategory, tool
from agent_service.tools.ideas.constants import (
    CHEAP_LMM_BATCH_THRESHOLD,
    DEFAULT_MAX_IDEAS,
    MAX_BRAINSTORM_TRIES,
    MIN_IDEAS,
    MIN_TOP_NEW_IDEA,
)
from agent_service.tools.ideas.prompts import (
    BRAINSTORM_IDEAS_DESCRIPTION,
    FINAL_IDEA_MAIN_PROMPT,
    FINAL_IDEA_SYS_PROMPT,
    IDEA_CLASSIFY_MAIN_PROMPT_STR,
    IDEA_CLASSIFY_SYS_PROMPT_STR,
    IDEA_CLUSTER_MAIN_PROMPT_STR,
    IDEA_CLUSTER_SYS_PROMPT_STR,
    IDEA_NOUN_MAIN_PROMPT,
    IDEA_REDUNDANCY_PASS_MAIN_PROMPT_STR,
    INITIAL_BRAINSTORM_PROMPT,
)
from agent_service.tools.ideas.utils import (
    create_small_text_groups,
    distinct_stock_count,
    distinct_text_count,
    get_source_texts,
    ideas_enabled,
)
from agent_service.tools.LLM_analysis.constants import MAX_CITATION_TRIES
from agent_service.tools.LLM_analysis.utils import extract_citations_from_gpt_output
from agent_service.tools.tool_log import tool_log
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency, identity
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.pagerduty import pager_wrapper
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.smart_classifier import SmartClassifier
from agent_service.utils.smart_clustering import SmartClustering
from agent_service.utils.string_utils import clean_to_json_if_needed
from agent_service.utils.text_utils import partition_to_smaller_text_sizes
from agent_service.utils.tool_diff import get_prev_run_info


async def initial_brainstorm(
    texts: List[Text],
    idea_definition: str,
    idea_noun: Dict[str, str],
    context: PlanRunContext,
    use_cheap: bool = True,
) -> List[Tuple[str, List[Text], int]]:
    logger = get_prefect_logger(__name__)
    text_group = TextGroup(val=texts)
    text_str = await Text.get_all_strs(
        text_group, include_header=True, include_symbols=True, text_group_numbering=True
    )

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    if use_cheap:
        llm = GPT(context=gpt_context, model=GPT4_O_MINI)
    else:
        llm = GPT(context=gpt_context, model=GPT4_O)

    # This usually isn't necessary but occasionally we have documents larger than the entire context window!
    text_str = GPTTokenizer(model=llm.model).do_truncation_if_needed(
        text_str,  # type: ignore
        [INITIAL_BRAINSTORM_PROMPT.template, idea_definition],
    )

    main_prompt = INITIAL_BRAINSTORM_PROMPT.format(
        idea_definition=idea_definition,
        idea=idea_noun["singular"],
        ideas=idea_noun["plural"],
        texts=text_str,
    )

    result = await llm.do_chat_w_sys_prompt(main_prompt, NO_PROMPT, output_json=True)

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
                    final_output.append((idea_dict[idea_noun["singular"]], sources, rank))  # type: ignore
            success = True
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Failed to parse correct json from initial brainstorm output: {result}")
            result = await llm.do_chat_w_sys_prompt(
                main_prompt, NO_PROMPT, output_json=True, no_cache=True
            )
            tries += 1

    return final_output


async def cluster_ideas(
    idea_definition: str,
    initial_ideas: List[Tuple[str, List[Text], int]],
    idea_noun: Dict[str, str],
    smart_pass: bool = False,
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

    if smart_pass:
        output_idea_clusters.sort(key=lambda x: len(x[0]), reverse=True)
        output_idea_clusters = output_idea_clusters[:50]  # keep this reasonably sized
        cluster_strs = []
        for i, cluster in enumerate(output_idea_clusters, start=1):
            cluster_strs.append(f"{i}. {cluster[0]}")
        prompt = Prompt(IDEA_REDUNDANCY_PASS_MAIN_PROMPT_STR, "IDEA_REDUNDANCY_PASS").format(
            idea_definition=idea_definition,
            idea=idea_noun["singular"],
            ideas=idea_noun["plural"],
            clusters="\n".join(cluster_strs),
        )
        result = await clusterer.gpt.do_chat_w_sys_prompt(prompt, NO_PROMPT)
        remove_cluster_idxs = set()
        for line in result.split("\n"):
            try:
                cluster_str, answer = line.strip().split("\t")
                curr_idx = int(cluster_str) - 1
            except ValueError:
                continue

            if answer == "No":
                remove_cluster_idxs.add(curr_idx)
            try:
                prev_idx = int(answer) - 1
                prev_cluster = output_idea_clusters[prev_idx]
                curr_cluster = output_idea_clusters[curr_idx]
                output_idea_clusters[prev_idx] = (
                    prev_cluster[0] + curr_cluster[0],
                    prev_cluster[1] + curr_cluster[1],
                    prev_cluster[2] + curr_cluster[2],
                )
                remove_cluster_idxs.add(curr_idx)
            except (ValueError, IndexError):
                continue

        output_idea_clusters = [
            cluster
            for i, cluster in enumerate(output_idea_clusters)
            if i not in remove_cluster_idxs
        ]

    return output_idea_clusters


async def classify_new_to_old_ideas(
    idea_definition: str, new_ideas: List[str], old_ideas: List[Idea]
) -> dict[str, Optional[Idea]]:
    classifier = SmartClassifier(
        identifier="IDEA_MATCH",
        items=idea_definition,
        sys_prompt=IDEA_CLASSIFY_SYS_PROMPT_STR,
        main_prompt=IDEA_CLASSIFY_MAIN_PROMPT_STR,
    )
    results: List[Optional[int]] = await classifier.apply_smart_classification(
        new_ideas, [idea.title for idea in old_ideas]
    )
    return {
        new_ideas[new_idx]: (old_ideas[old_idx] if old_idx is not None else None)
        for new_idx, old_idx in enumerate(results)
    }


async def create_final_idea(
    idea_definition: str,
    idea_formulations: List[str],
    idea_relevant_texts: List[Text],
    idea_noun: Dict[str, str],
    context: PlanRunContext,
) -> Idea:
    logger = get_prefect_logger(__name__)

    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)

    text_group = TextGroup(val=idea_relevant_texts)
    text_str: str = await Text.get_all_strs(  # type: ignore
        text_group, include_header=True, include_symbols=True, text_group_numbering=True
    )
    ideas_str = "\n".join(idea_formulations)

    text_str = GPTTokenizer(GPT4_O).do_truncation_if_needed(
        text_str,
        [
            FINAL_IDEA_MAIN_PROMPT.template,
            FINAL_IDEA_SYS_PROMPT.template,
            ideas_str,
            idea_definition,
        ],
    )

    idea_sg = idea_noun["singular"]
    idea_pl = idea_noun["plural"]

    main_prompt = FINAL_IDEA_MAIN_PROMPT.format(
        idea_definition=idea_definition,
        initial_ideas=ideas_str,
        source_documents=text_str,
        idea=idea_sg,
        ideas=idea_pl,
    )
    sys_prompt = FINAL_IDEA_SYS_PROMPT.format(idea=idea_sg, ideas=idea_pl)

    result = await llm.do_chat_w_sys_prompt(main_prompt, sys_prompt)

    lines = result.strip().split("\n")
    title = lines[0].strip("# ")
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


async def get_idea_noun(idea_definition: str, context: PlanRunContext) -> Dict[str, str]:
    gpt_context = create_gpt_context(
        GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
    )
    llm = GPT(context=gpt_context, model=GPT4_O)
    result = await llm.do_chat_w_sys_prompt(
        IDEA_NOUN_MAIN_PROMPT.format(idea_definition=idea_definition), NO_PROMPT
    )
    return json.loads(clean_to_json_if_needed(result))


class BrainstormIdeasFromTextsInput(ToolArgs):
    idea_definition: str
    texts: List[Text]
    max_ideas: int = DEFAULT_MAX_IDEAS


@tool(
    description=BRAINSTORM_IDEAS_DESCRIPTION,
    category=ToolCategory.IDEAS,
    enabled=True,
    enabled_checker_func=ideas_enabled,
)
async def brainstorm_ideas_from_text(
    args: BrainstormIdeasFromTextsInput, context: PlanRunContext
) -> List[Idea]:
    logger = get_prefect_logger(__name__)
    if len(args.texts) == 0:
        raise EmptyInputError("Missing input texts to idea brainstorm")

    partitioned_texts: List[Text] = await partition_to_smaller_text_sizes(args.texts, context)

    prev_texts: Optional[List[Text]] = None
    prev_output: Optional[List[Idea]] = None

    try:  # since everything associated with diffing is optional, put in try/except
        # Update mode
        prev_run_info = await get_prev_run_info(context, "brainstorm_ideas_from_text")
        if prev_run_info is not None:
            prev_args = BrainstormIdeasFromTextsInput.model_validate_json(prev_run_info.inputs_str)
            prev_texts = await partition_to_smaller_text_sizes(prev_args.texts, context)
            prev_output = prev_run_info.output  # type:ignore
            if prev_output is not None and len(set(prev_texts) | set(partitioned_texts)) == len(
                prev_texts
            ):
                # if exactly the same texts, just return the old output
                return prev_output
    except Exception as e:
        logger.exception(
            f"Failed attempt to update from previous iteration due to {e}, from scratch fallback"
        )
        pager_wrapper(
            current_frame=inspect.currentframe(),
            module_name=__name__,
            context=context,
            e=e,
            classt="AgentUpdateError",
            summary="Failed to get previous run info or to partition into smaller texts",
        )

    idea_noun = await get_idea_noun(args.idea_definition, context)

    text_groups = await create_small_text_groups(partitioned_texts)

    if len(text_groups) > CHEAP_LMM_BATCH_THRESHOLD:
        # if we are reading a lot of text, use the cheap LLM
        use_cheap = True
    else:
        use_cheap = False
    tasks = []
    # do brainstorming for each group that can fit in a single GPT context call
    for text_group in text_groups:
        tasks.append(
            initial_brainstorm(
                text_group, args.idea_definition, idea_noun, context, use_cheap=use_cheap
            )
        )
    result = await gather_with_concurrency(tasks)
    initial_ideas = [idea for idea_groups in result for idea in idea_groups]

    if len(initial_ideas) < 2:
        raise EmptyOutputError("Failed to brainstorm ideas from provided texts")

    # cluster ideas across the runs
    clustered_ideas = await cluster_ideas(
        args.idea_definition, initial_ideas, idea_noun, smart_pass=use_cheap
    )
    # only include ideas with at least two citations provided there are at least 5 ideas with 2 citations
    two_citation_clustered_ideas = [idea for idea in clustered_ideas if len(idea[1]) > 1]
    if len(two_citation_clustered_ideas) >= MIN_IDEAS:
        clustered_ideas = two_citation_clustered_ideas
    # sort ideas by distinct relevent stocks, then distinct texts plus inverse rank and chop to max
    clustered_ideas.sort(
        key=lambda x: (distinct_stock_count(x[1]), distinct_text_count(x[1]) + x[2]), reverse=True
    )

    if prev_output and prev_texts:  # updating
        curr_texts = set(partitioned_texts)
        new_texts = curr_texts - set(prev_texts)
        new_ideas_strs = [str(idea_formulations) for idea_formulations, _, _ in clustered_ideas]

        new_idea_mapping = await classify_new_to_old_ideas(
            args.idea_definition, new_ideas_strs, prev_output
        )
        final_tasks: List[Coroutine] = []
        used_old_ideas: Set[Idea] = set()  # just in case a new idea gets mapped to multiple old one
        replace_dict = {}  # store old idea names to use
        # Basic idea here: We iterate over the new ideas generated in this pass, for the top few
        # we include them even if we don't have a mapping to the old idea (which means they'll be
        # totally new and will require new stuff downstream). After that point, we try only to include
        # new ideas that are also old ideas, to avoid major shifts in ideas from run to run, though we
        # may end up rewriting their description if indicated. If there's space left, include old
        # ideas that haven't changed
        i = 0
        while i < len(clustered_ideas) and len(final_tasks) < args.max_ideas:
            idea_formulations, idea_relevant_texts, _ = clustered_ideas[i]
            if new_idea_mapping[str(idea_formulations)] is None:
                if len(final_tasks) < MIN_TOP_NEW_IDEA:
                    final_tasks.append(
                        create_final_idea(
                            args.idea_definition,
                            idea_formulations,
                            idea_relevant_texts,
                            idea_noun,
                            context,
                        )
                    )
                else:  # low ranked new idea that doesn't have old idea, skip
                    pass
            else:
                old_idea = new_idea_mapping[str(idea_formulations)]
                if old_idea is None or old_idea in used_old_ideas:
                    i += 1
                    continue  # can't add the same idea twice
                old_source_texts = get_source_texts(old_idea)

                # we rewrite the idea if there are new relevant sources OR old sources are no longer in our input
                if any(
                    [idea_relevant_text in new_texts for idea_relevant_text in idea_relevant_texts]
                ) or any(
                    [old_source_text not in curr_texts for old_source_text in old_source_texts]
                ):
                    replace_dict[len(final_tasks)] = old_idea
                    final_tasks.append(
                        create_final_idea(
                            args.idea_definition,
                            idea_formulations,
                            idea_relevant_texts,
                            idea_noun,
                            context,
                        )
                    )
                    used_old_ideas.add(old_idea)
                else:  # can just use the old idea as is
                    final_tasks.append(identity(old_idea))
                    used_old_ideas.add(old_idea)

            i += 1

        final_ideas = await gather_with_concurrency(final_tasks, n=len(clustered_ideas))
        for i, idea in replace_dict.items():
            final_ideas[i].title = idea.title  # use the old title so it is considered same idea

        # if we've still got space, add in old ideas that haven't been used and still have all their citations
        i = 0
        while len(final_ideas) < args.max_ideas and i < len(prev_output):
            old_idea = prev_output[i]
            if old_idea not in used_old_ideas:
                old_source_texts = get_source_texts(old_idea)
                if not any(
                    [old_source_text not in curr_texts for old_source_text in old_source_texts]
                ):
                    final_ideas.append(old_idea)
            i += 1

    else:
        # generate final ideas with references
        clustered_ideas = clustered_ideas[: args.max_ideas]
        final_tasks = [
            create_final_idea(
                args.idea_definition, idea_formulations, idea_relevant_texts, idea_noun, context
            )
            for idea_formulations, idea_relevant_texts, _ in clustered_ideas
        ]
        final_ideas = await gather_with_concurrency(final_tasks, n=len(clustered_ideas))

    await tool_log(
        f"Brainstormed {len(final_ideas)} ideas: {"; ".join([idea.title for idea in final_ideas])}",
        context,
    )

    return final_ideas
