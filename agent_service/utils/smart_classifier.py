import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from agent_service.GPT.constants import FILTER_CONCURRENCY, GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.prompt_utils import Prompt

logger = logging.getLogger(__name__)


def clustering_to_classifier(prompt: str) -> str:
    return prompt.replace("text1", "instance").replace("text2", "label")


class SmartClassifier:
    def __init__(
        self,
        identifier: str,
        items: str,
        sys_prompt: str,
        main_prompt: str,
        model: str = GPT4_O,
        check_n: int = 1,
        context: Optional[Dict[str, str]] = None,
    ) -> None:
        # sys_prompt is the system prompt sent to GPT to instruct the model on how
        # it should behave
        # main_prompt is the main prompt sent to GPT containing the instructions is is
        # expected to carry out
        #
        # the main prompt should have an instance and label, asks if instance is a case of label
        # check_n > 1 means we will check more than one label (n nearest label)

        self.gpt = GPT(model=model, context=context)
        self.model = model
        self.items = items
        self.sys_prompt = Prompt(sys_prompt, identifier + "_SMART_CLASSIFER_SYS_PROMPT")
        self.main_prompt = Prompt(main_prompt, identifier + "_SMART_CLASSIFIER_MAIN_PROMPT")
        self.check_n = check_n

    async def apply_smart_classification(
        self, instances: List[str], labels: List[str]
    ) -> List[Optional[int]]:
        # Ouput is the same length as instance, picks a label idx (or None) using a mixture of embeddings and GPT
        triples = await self._get_nearest_label_idxs(instances, labels)
        is_match_list = await self._check_pairs(instances, labels, triples)
        label_idx_list = self._finalize_classification(len(instances), triples, is_match_list)
        return label_idx_list

    async def _get_text_embeddings(self, text_sources: List[str]) -> np.ndarray:
        # get embeddings for all texts, return an array of size number of text sources X embedding size
        tasks = [self.gpt.embed_text(text_source) for text_source in text_sources]
        embeddings = await gather_with_concurrency(tasks, FILTER_CONCURRENCY)
        return np.array(embeddings)

    async def _get_nearest_label_idxs(
        self, instances: List[str], labels: List[str]
    ) -> List[Tuple[int, int, int]]:
        # get (i, j, k) triples where label k is the ith closest to instance j.
        instance_embeddings = await self._get_text_embeddings(instances)
        label_embeddings = await self._get_text_embeddings(labels)
        sims = -cdist(instance_embeddings, label_embeddings, metric="cosine")
        sim_args = np.argsort(sims, axis=1)  # closest label indices at the end
        nearest_triples = []
        num_to_check = min(self.check_n, len(labels))
        for j in range(len(sims)):  # For each instance
            for i in range(1, num_to_check + 1):
                nearest_triples.append((i, j, sim_args[j, -i]))
        return nearest_triples

    async def _check_pairs(
        self, instances: List[str], labels: List[str], triples: List[Tuple[int, int, int]]
    ) -> List[bool]:
        # For the triple of indices where the second and third correspond to instances/labels
        # check to see if there is a match, return
        tasks = []
        for _, instance_idx, label_idx in triples:
            tasks.append(self._check_pair(instances[instance_idx], labels[label_idx]))
        concurrency = FILTER_CONCURRENCY
        return await gather_with_concurrency(tasks, concurrency)

    async def _check_pair(self, instance: str, label: str) -> bool:
        # Ask GPT if the pairs of indices corresponding to an instance/label text pair match
        result = await self.gpt.do_chat_w_sys_prompt(
            self.main_prompt.format(instance=instance, label=label),
            self.sys_prompt.format(items=self.items),
            max_tokens=2,
        )

        return result.lower().startswith("yes")

    def _finalize_classification(
        self, num_instances: int, triples: List[Tuple[int, int, int]], is_match_list: List[bool]
    ) -> List[Optional[int]]:
        label_idxs: List[Optional[int]] = [None] * num_instances
        for (_, instance_idx, label_idx), is_match in zip(triples, is_match_list):
            if is_match and label_idxs[instance_idx] is None:
                # this relies on the original ordering of the triples, nearest idx first
                label_idxs[instance_idx] = label_idx
        return label_idxs
