import enum

from pydantic import BaseModel


class HorizonTabs(enum.StrEnum):
    CS = "CS"
    ENG = "ENG"
    PROD = "PROD"


class HorizonUser(BaseModel):
    userId: str
    name: str
    userType: HorizonTabs
