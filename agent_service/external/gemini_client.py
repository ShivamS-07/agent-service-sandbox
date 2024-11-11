from enum import StrEnum
from typing import Dict, Optional

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Tool,
    grounding,
)

PROJECT_ID = "gemini-project-441015"
DEFAULT_TEMPERATURE = 0.0

vertexai.init(project=PROJECT_ID)


class GoogleAIModel(StrEnum):
    GEMINI_FLASH = "gemini-1.5-flash"
    GEMINI_PRO = "gemini-1.5-pro"


class GeminiClient:
    def __init__(
        self,
        model: GoogleAIModel = GoogleAIModel.GEMINI_FLASH,
        temperature: Optional[float] = None,
        context: Optional[Dict[str, str]] = None,
    ):
        self.model = GenerativeModel(model)
        self.temperature = temperature
        self.tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
        self.context = context or {}

    def _get_temperature(self, temperature: Optional[float] = None) -> float:
        if temperature is not None:
            return temperature
        elif self.temperature is not None:
            return self.temperature
        return DEFAULT_TEMPERATURE

    async def query_google_search(self, query: str, temperature: Optional[float] = None) -> str:
        response = await self.model.generate_content_async(
            query,
            tools=[self.tool],
            generation_config=GenerationConfig(temperature=self._get_temperature(temperature)),
            labels=self.context,
        )
        return response.text
