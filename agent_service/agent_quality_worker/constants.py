from typing import Dict

from agent_service.agent_quality_worker.models import HorizonTabs, HorizonUser

CS_TIERED_ASSIGNMENT_ALLOCATIONS = {
    1: 0.8,
    2: 0.2,
}  # Allocations should add up to 1.0 (100%)

HORIZON_USERS_PROD: Dict[str, HorizonUser] = {
    # PROD - CS
    # T1
    "5920388a-717a-42b0-9cec-2e999faa6b9e": HorizonUser(
        userId="5920388a-717a-42b0-9cec-2e999faa6b9e",
        name="Crystal",
        userType=HorizonTabs.CS,
        tier=1,
    ),
    "00a4c63d-99b2-478f-aa27-a0d263e3e0ba": HorizonUser(
        userId="00a4c63d-99b2-478f-aa27-a0d263e3e0ba",
        name="Emmalyn",
        userType=HorizonTabs.CS,
        tier=1,
    ),
    "2bcaa299-a84f-46ef-8b19-452559845f73": HorizonUser(
        userId="2bcaa299-a84f-46ef-8b19-452559845f73",
        name="George",
        userType=HorizonTabs.CS,
        tier=1,
    ),
    "7879ce5a-bee2-469b-9676-d3c5bb9ed804": HorizonUser(
        userId="7879ce5a-bee2-469b-9676-d3c5bb9ed804",
        name="Hayley",
        userType=HorizonTabs.CS,
        tier=1,
    ),
    "72c8f2ba-fab8-439a-a94d-b725cc485b76": HorizonUser(
        userId="72c8f2ba-fab8-439a-a94d-b725cc485b76",
        name="Davison",
        userType=HorizonTabs.CS,
        tier=1,
    ),
    "75916863-aba9-4d68-877d-34d224dbb4bf": HorizonUser(
        userId="75916863-aba9-4d68-877d-34d224dbb4bf",
        name="Jamie",
        userType=HorizonTabs.CS,
        tier=1,
    ),
    "e5f85515-c073-433b-bad2-f05fa3cb38c3": HorizonUser(
        userId="e5f85515-c073-433b-bad2-f05fa3cb38c3",
        name="Jordan",
        userType=HorizonTabs.CS,
        tier=2,
    ),
    "2e79d407-b5e0-45d9-8a18-06935a56c3c1": HorizonUser(
        userId="2e79d407-b5e0-45d9-8a18-06935a56c3c1",
        name="Songjia",
        userType=HorizonTabs.CS,
        tier=1,
    ),
    "85f9a052-e7c3-45c1-a1b4-eee8d7cb6322": HorizonUser(
        userId="85f9a052-e7c3-45c1-a1b4-eee8d7cb6322",
        name="Maria",
        userType=HorizonTabs.CS,
        tier=1,
    ),
    # T2
    "7bc6346f-41f7-407e-9b11-7d9cc40d4790": HorizonUser(
        userId="7bc6346f-41f7-407e-9b11-7d9cc40d4790",
        name="Josh F",
        userType=HorizonTabs.CS,
        tier=2,
    ),
    "86de3356-ed6b-4228-afac-f57781268e1f": HorizonUser(
        userId="86de3356-ed6b-4228-afac-f57781268e1f",
        name="Hogan",
        userType=HorizonTabs.CS,
        tier=2,
    ),
    "89f6c72f-d9b8-40d0-926b-d958841cc906": HorizonUser(
        userId="89f6c72f-d9b8-40d0-926b-d958841cc906",
        name="Porter",
        userType=HorizonTabs.CS,
        tier=2,
    ),
    "28d8413a-5d73-4051-b208-32c82d41a5c4": HorizonUser(
        userId="28d8413a-5d73-4051-b208-32c82d41a5c4",
        name="Jan",
        userType=HorizonTabs.CS,
        tier=2,
    ),
    # PROD - ENG
    "a2a77e4d-15b6-4723-b36a-af5c90e1c09c": HorizonUser(
        userId="a2a77e4d-15b6-4723-b36a-af5c90e1c09c",
        name="Simon",
        userType=HorizonTabs.ENG,
    ),
    "87f5b738-4ef8-4f3b-b038-1ba169ded59d": HorizonUser(
        userId="87f5b738-4ef8-4f3b-b038-1ba169ded59d",
        name="Julian",
        userType=HorizonTabs.ENG,
    ),
    "67fd46ff-cb0b-4ffe-9ba5-c52f8b706063": HorizonUser(
        userId="67fd46ff-cb0b-4ffe-9ba5-c52f8b706063",
        name="Richard",
        userType=HorizonTabs.ENG,
    ),
    "c1874748-2d38-4bed-9b34-20081e09adc3": HorizonUser(
        userId="c1874748-2d38-4bed-9b34-20081e09adc3",
        name="Jackson",
        userType=HorizonTabs.ENG,
    ),
    # PROD - PRODUCT
    "3726119c-92cd-4c5f-97c5-23ed908e78be": HorizonUser(
        userId="3726119c-92cd-4c5f-97c5-23ed908e78be",
        name="Alex",
        userType=HorizonTabs.PROD,
    ),
}

HORIZON_USERS_DEV: Dict[str, HorizonUser] = {
    # DEV - CS (for testing)
    "3a2eaf66-3d4f-4f9f-b9eb-dbe15972c894": HorizonUser(
        userId="3a2eaf66-3d4f-4f9f-b9eb-dbe15972c894",
        name="Simon",
        userType=HorizonTabs.CS,
    ),
    "3fa644a2-7b02-4c0e-af12-166add8da0ad": HorizonUser(
        userId="3fa644a2-7b02-4c0e-af12-166add8da0ad",
        name="simon-test-user",
        userType=HorizonTabs.CS,
    ),
    "2ab84663-ee32-45e4-922b-82e0f089aab2": HorizonUser(
        userId="2ab84663-ee32-45e4-922b-82e0f089aab2",
        name="Richard",
        userType=HorizonTabs.CS,
    ),
    # DEV - ENG
    "a5d534c9-5426-4387-a298-723c5e09ecab": HorizonUser(
        userId="a5d534c9-5426-4387-a298-723c5e09ecab",
        name="William",
        userType=HorizonTabs.ENG,
    ),
    "87f5b738-4ef8-4f3b-b038-1ba169ded59d": HorizonUser(
        userId="87f5b738-4ef8-4f3b-b038-1ba169ded59d",
        name="Julian",
        userType=HorizonTabs.ENG,
    ),
    "3de70fb5-9c1d-442b-b12d-d218d70ab1b5": HorizonUser(
        userId="3de70fb5-9c1d-442b-b12d-d218d70ab1b5",
        name="Richard",
        userType=HorizonTabs.ENG,
    ),
    "911139b0-0e1a-4799-b799-29cb8d568bc2": HorizonUser(
        userId="911139b0-0e1a-4799-b799-29cb8d568bc2",
        name="Jackson",
        userType=HorizonTabs.ENG,
    ),
    # DEV - PRODUCT
    "3726119c-92cd-4c5f-97c5-23ed908e78be": HorizonUser(
        userId="3726119c-92cd-4c5f-97c5-23ed908e78be",
        name="Alex",
        userType=HorizonTabs.PROD,
    ),
    "c47aa9c0-9cc3-4b5c-bd48-6fd43d9a4f49": HorizonUser(
        userId="c47aa9c0-9cc3-4b5c-bd48-6fd43d9a4f49",
        name="Stu",
        userType=HorizonTabs.PROD,
    ),
    "661f2859-8f28-4b9a-b2bc-c34328e73af5": HorizonUser(
        userId="661f2859-8f28-4b9a-b2bc-c34328e73af5",
        name="April",
        userType=HorizonTabs.PROD,
    ),
    "d5246e24-ffb5-4a47-8d99-f18d50fc8ff3": HorizonUser(
        userId="d5246e24-ffb5-4a47-8d99-f18d50fc8ff3",
        name="Nick",
        userType=HorizonTabs.PROD,
    ),
}


JIRA_USER = "project.management@gradientboostedinvestments.com"
JIRA_URL = "https://gradientboostedinvestments.atlassian.net"
CS_REVIEWER_COLUMN = "cs_reviewer"
ENG_REVIEWER_COLUMN = "eng_reviewer"
PROD_REVIEWER_COLUMN = "prod_reviewer"
MAX_REVIEWS = 150
