import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    template_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    prompt: str
    category: str
    created_at: datetime.datetime
