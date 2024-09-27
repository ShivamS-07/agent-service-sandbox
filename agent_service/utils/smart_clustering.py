from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

from agent_service.GPT.constants import FILTER_CONCURRENCY, GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.prefect import get_prefect_logger
from agent_service.utils.prompt_utils import Prompt

MAX_PAIR_PER_ROUND = 20


def flatten_concurrency_output(L_of_L: List[List[Any]]) -> List[Any]:
    return [item for L in L_of_L for item in L]


def get_equal_sized_breakdown(total_num: int, max_run_size: int) -> Tuple[int, int]:
    if total_num % max_run_size == 0:
        return total_num // max_run_size, max_run_size
    else:
        num_it = total_num // max_run_size + 1
        num_per_run = total_num // (num_it) + 1
        return num_it, num_per_run


def get_groups_from_pairs(same_category_pairs: List[Tuple[int, int]]) -> List[List[int]]:
    # Use transitivity to go from confirmed same-category pairs to clusters corresponding
    # to groups of the same category. Basically, just create groups for pairs and then
    # join groups when a confirmed same_category_pair are in the different groups
    to_group_lookup = {}
    for num1, num2 in same_category_pairs:
        if num1 not in to_group_lookup and num2 not in to_group_lookup:
            group = set([num1, num2])
            to_group_lookup[num1] = group
            to_group_lookup[num2] = group
        elif num2 not in to_group_lookup:
            to_group_lookup[num1].add(num2)
            to_group_lookup[num2] = to_group_lookup[num1]
        elif num1 not in to_group_lookup:
            to_group_lookup[num2].add(num1)
            to_group_lookup[num1] = to_group_lookup[num2]
        elif to_group_lookup[num1] != to_group_lookup[num2]:
            to_group_lookup[num1].update(to_group_lookup[num2])
            for num in to_group_lookup[num2]:
                to_group_lookup[num] = to_group_lookup[num1]

    return [list(group) for group in set([tuple(group) for group in to_group_lookup.values()])]


class SmartClustering:
    def __init__(
        self,
        identifier: str,
        items: str,
        sys_prompt: str,
        main_prompt: str,
        model: str = GPT4_O,
        context: Optional[Dict[str, str]] = None,
        max_pair_per_round: int = MAX_PAIR_PER_ROUND,
        single_pair_check: bool = False,
    ) -> None:
        # sys_prompt is the system prompt sent to GPT to instruct the model on how
        # it should behave
        # main_prompt is the main prompt sent to GPT containing the instructions is is
        # expected to carry out
        #
        # the main prompt should end with "\n\n{text_pairs}:" which will allow SmartClustering
        # to properly pass in the pairs of data into the GPT message
        #
        # ie. "Decide whether or not the following pairs ... Here are the pairs of headlines:\n\n{article_pairs}"
        #

        self.gpt = GPT(model=model, context=context)
        self.sys_prompt = Prompt(sys_prompt, identifier + "_SMART_CLUSTERING_SYS_PROMPT")
        self.main_prompt = Prompt(main_prompt, identifier + "_SMART_CLUSTERING_MAIN_PROMPT")
        self.items = items
        self.model = model
        self.single_pair_check = single_pair_check

        self.max_pair_per_round = max_pair_per_round

    async def apply_smart_clustering(self, text_sources: List[str]) -> List[List[int]]:
        # Clusters the input text_sources using a mixture of embeddings and GPT
        pairs = await self._get_nearest_pairs(text_sources)
        if self.single_pair_check:
            same_category_pairs = await self._check_pairs_single(text_sources, pairs)
        else:
            same_category_pairs = await self._check_pairs_batch(text_sources, pairs)
        groups = get_groups_from_pairs(same_category_pairs)
        return groups

    async def _get_text_embeddings(self, text_sources: List[str]) -> np.ndarray:
        # get embeddings for all text sources, return an array of size number of text sources X embedding size
        tasks = [self.gpt.embed_text(text_source) for text_source in text_sources]
        embeddings = await gather_with_concurrency(tasks, FILTER_CONCURRENCY)
        return np.array(embeddings)

    async def _get_nearest_pairs(self, text_sources: List[str]) -> Set[Tuple[int, int]]:
        # get pairs of indices (ints) of text_sources which are closest in embedding
        # space. A new pair is added for each text source in text_sources, which means if a pair
        # is already in the list, the next closest text in text_sources is paired, etc.
        embeddings = await self._get_text_embeddings(text_sources)
        sims = -squareform(
            pdist(embeddings, metric="cosine")
        )  # gives distances, swap sign to get similarities
        sim_args = np.argsort(sims, axis=1)  # closest text_source indices at the end
        nearest_pairs = set()
        for i in range(len(sims)):  # For each text_source in text_sources
            j = 2  # last one is itself, so start with 2
            while j < len(sim_args) + 1:  # just to be sure, shouldn't happen
                # get the index pair for this text_source (i) and the closest text_source
                # that we haven't looked at sim_args[i, -j], sort so we can
                # match duplicates
                pair = tuple(sorted([sim_args[i, -j], i]))
                if pair not in nearest_pairs:  # if we haven't seen this pair before
                    nearest_pairs.add(pair)
                    break  # we are done with this text_source
                j += 1  # otherwise keep going until we find a new pair
        return nearest_pairs

    async def _check_pairs_batch(
        self, text_sources: List[str], pairs: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        # Check if the pairs of indices corresponding to pairs of text_sources formed from text_sources
        # in fact refer to the same category, return all those that do
        num_iter, max_size = get_equal_sized_breakdown(len(pairs), self.max_pair_per_round)
        pairs = list(pairs)
        tasks = []
        for i in range(num_iter):
            tasks.append(
                self._pair_run_batch(text_sources, pairs[i * max_size : (i + 1) * max_size])
            )
        results = await gather_with_concurrency(tasks, FILTER_CONCURRENCY)
        output_pairs = flatten_concurrency_output(results)
        return output_pairs

    async def _pair_run_batch(
        self, text_sources: List[str], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        # Ask GPT if the pairs of indices corresponding to pairs of text from text_sources
        # actually correspond to the same category, return them if so
        logger = get_prefect_logger(__name__)
        text_pairs = self._get_text_pair_list(text_sources, pairs)
        output_pairs = []
        result = await self.gpt.do_chat_w_sys_prompt(
            self.main_prompt.format(text_pairs=text_pairs), self.sys_prompt.format(items=self.items)
        )
        for line in result.split("\n"):
            if not line.strip():
                continue
            try:
                num, answer = line.strip().split(" ")
                if answer == "Yes":
                    output_pairs.append(pairs[int(num) - 1])
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"Failed while parsing smart clustering GPT output: line: {line}, error:{e}"
                )

        return output_pairs

    def _get_text_pair_list(self, text_sources: List[str], pairs: List[Tuple[int, int]]) -> str:
        # creates a numbered list of pairs of text sources, pairs is a list of indices
        return "\n\n".join(
            [
                f"{num + 1}\n{text_sources[art_num1]}\n{text_sources[art_num2]}"
                for num, (art_num1, art_num2) in enumerate(pairs)
            ]
        )

    async def _check_pairs_single(
        self, text_sources: List[str], pairs: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        # Check if the pairs of indices corresponding to pairs of text_sources formed from text_sources
        # in fact refer to the same category, return all those that do
        pairs = list(pairs)
        tasks = []
        for pair in pairs:
            tasks.append(
                self.gpt.do_chat_w_sys_prompt(
                    self.main_prompt.format(
                        text1=text_sources[pair[0]], text2=text_sources[pair[1]]
                    ),
                    self.sys_prompt.format(items=self.items),
                    max_tokens=2,
                )
            )
        results = await gather_with_concurrency(tasks, FILTER_CONCURRENCY)
        output_pairs = [
            pair for pair, result in zip(pairs, results) if result.lower().startswith("yes")
        ]
        return output_pairs
