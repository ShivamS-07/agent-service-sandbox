from collections import defaultdict
from enum import StrEnum
from typing import Dict, Optional, cast

import google.generativeai as genai
from gbi_common_py_utils.utils.environment import (
    PROD_TAG,
    STAGING_TAG,
    get_environment_tag,
)
from gbi_common_py_utils.utils.ssm import get_param

from agent_service.io_type_utils import Citation, HistoryEntry
from agent_service.io_types.text import (
    GoogleGroundingSnippetText,
    Text,
    TextCitation,
    TextOutput,
)
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.postgres import SyncBoostedPG

DEFAULT_TEMPERATURE = 0.0


class GoogleAIModel(StrEnum):
    GEMINI_FLASH = "gemini-1.5-flash"
    GEMINI_PRO = "gemini-1.5-pro"


api_key = (
    get_param(name="/google/gemini/api-key")
    if get_environment_tag() in (STAGING_TAG, PROD_TAG)
    else get_param("/google/gemini/api-key-dev")
)
genai.configure(api_key=api_key)


class GeminiClient:
    def __init__(
        self,
        model: GoogleAIModel = GoogleAIModel.GEMINI_FLASH,
        temperature: Optional[float] = None,
        context: Optional[Dict[str, str]] = None,
    ):
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature
        self.context = context or {}

    def _get_temperature(self, temperature: Optional[float] = None) -> float:
        if temperature is not None:
            return temperature
        elif self.temperature is not None:
            return self.temperature
        return DEFAULT_TEMPERATURE

    async def query_google_grounding(
        self, query: str, temperature: Optional[float] = None, db: Optional[BoostedPG] = None
    ) -> TextOutput:
        response = await self.model.generate_content_async(
            query,
            tools="google_search_retrieval",
            generation_config=genai.GenerationConfig(
                temperature=self._get_temperature(temperature)
            ),
        )
        grounding_metadata = response.candidates[0].grounding_metadata
        text = Text(val=response.text)
        citations: list[Citation] = []
        chunk_index_text_offset_map: Dict[int, list[int]] = defaultdict(list)
        # Map chunks to their inline indexes, and then create citations for them
        for support in grounding_metadata.grounding_supports:
            for index in support.grounding_chunk_indices:
                chunk_index_text_offset_map[index].append(max(support.segment.end_index - 1, 0))
        for i, chunk in enumerate(grounding_metadata.grounding_chunks):
            inline_offsets = chunk_index_text_offset_map.get(i)
            if not inline_offsets:
                citations.append(
                    TextCitation(
                        source_text=GoogleGroundingSnippetText(
                            title=chunk.web.title, url=chunk.web.uri
                        ),
                    )
                )
            else:
                for offset in inline_offsets:
                    citations.append(
                        TextCitation(
                            source_text=GoogleGroundingSnippetText(
                                title=chunk.web.title, url=chunk.web.uri
                            ),
                            citation_text_offset=offset,
                        )
                    )
        if citations:
            text.history.append(HistoryEntry(citations=citations))

        return cast(TextOutput, await text.to_rich_output(pg=db or SyncBoostedPG()))
