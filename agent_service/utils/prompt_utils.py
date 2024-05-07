import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class FilledPrompt:
    filled_prompt: str
    template: Optional[str] = None
    name: Optional[str] = None
    template_args: Optional[Dict[str, Any]] = None


@dataclass
class Prompt:
    template: str
    name: str

    def format(self, **kwargs) -> FilledPrompt:  # type: ignore
        filled_prompt = self.template.format(**kwargs)
        return FilledPrompt(
            filled_prompt=filled_prompt,
            template=self.template,
            name=self.name,
            template_args=kwargs,
        )
