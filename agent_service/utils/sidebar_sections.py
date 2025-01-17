import uuid
from typing import List, Optional

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from agent_service.utils.date_utils import get_now_utc


class SidebarSection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    created_at: str = Field(default_factory=lambda: get_now_utc().isoformat())
    updated_at: str = Field(default_factory=lambda: get_now_utc().isoformat())

    # Only populated when sending to FE
    index: Optional[int] = None


def find_sidebar_section(sections: List[SidebarSection], section_id: str) -> int:
    index = -1
    for i, section in enumerate(sections):
        if section.id == section_id:
            index = i
            break
    if index == -1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Section not found")
    return index
