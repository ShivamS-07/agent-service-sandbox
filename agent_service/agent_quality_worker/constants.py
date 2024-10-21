from typing import Dict

from agent_service.agent_quality_worker.models import HorizonTabs, HorizonUser

HORIZON_USERS_PROD: Dict[str, HorizonUser] = {
    # PROD - CS
    "00a4c63d-99b2-478f-aa27-a0d263e3e0ba": HorizonUser(
        userId="00a4c63d-99b2-478f-aa27-a0d263e3e0ba",
        name="Emmalyn",
        userType=HorizonTabs.CS,
    ),
    "25b12e24-f0a0-48e5-9ad2-a60f323b3f68": HorizonUser(
        userId="25b12e24-f0a0-48e5-9ad2-a60f323b3f68",
        name="Peter",
        userType=HorizonTabs.CS,
    ),
    "2bcaa299-a84f-46ef-8b19-452559845f73": HorizonUser(
        userId="2bcaa299-a84f-46ef-8b19-452559845f73",
        name="George",
        userType=HorizonTabs.CS,
    ),
    "7879ce5a-bee2-469b-9676-d3c5bb9ed804": HorizonUser(
        userId="7879ce5a-bee2-469b-9676-d3c5bb9ed804",
        name="Hayley",
        userType=HorizonTabs.CS,
    ),
    "72c8f2ba-fab8-439a-a94d-b725cc485b76": HorizonUser(
        userId="72c8f2ba-fab8-439a-a94d-b725cc485b76",
        name="Davison",
        userType=HorizonTabs.CS,
    ),
    "75916863-aba9-4d68-877d-34d224dbb4bf": HorizonUser(
        userId="75916863-aba9-4d68-877d-34d224dbb4bf",
        name="Jamie",
        userType=HorizonTabs.CS,
    ),
    "2e79d407-b5e0-45d9-8a18-06935a56c3c1": HorizonUser(
        userId="2e79d407-b5e0-45d9-8a18-06935a56c3c1",
        name="Songjia",
        userType=HorizonTabs.CS,
    ),
    "85f9a052-e7c3-45c1-a1b4-eee8d7cb6322": HorizonUser(
        userId="85f9a052-e7c3-45c1-a1b4-eee8d7cb6322",
        name="Maria",
        userType=HorizonTabs.CS,
    ),
    # PROD - ENG
    "a2a77e4d-15b6-4723-b36a-af5c90e1c09c": HorizonUser(
        userId="a2a77e4d-15b6-4723-b36a-af5c90e1c09c",
        name="Simon",
        userType=HorizonTabs.ENG,
    ),
    "f6fe6a54-c15c-4893-9909-90657be7f19f": HorizonUser(
        userId="f6fe6a54-c15c-4893-9909-90657be7f19f",
        name="David",
        userType=HorizonTabs.ENG,
    ),
    "7f08e5de-e217-41a3-bf7e-2dc7ce4c9f05": HorizonUser(
        userId="7f08e5de-e217-41a3-bf7e-2dc7ce4c9f05",
        name="William",
        userType=HorizonTabs.ENG,
    ),
    "514e30db-054c-4ead-b105-98456eef18e1": HorizonUser(
        userId="514e30db-054c-4ead-b105-98456eef18e1",
        name="Mazin",
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
    "1c293abd-2485-48b9-b78a-6094f4ef6c5b": HorizonUser(
        userId="1c293abd-2485-48b9-b78a-6094f4ef6c5b",
        name="April",
        userType=HorizonTabs.PROD,
    ),
    "d5246e24-ffb5-4a47-8d99-f18d50fc8ff3": HorizonUser(
        userId="d5246e24-ffb5-4a47-8d99-f18d50fc8ff3",
        name="Nick",
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
    # DEV - ENG
    "f6fe6a54-c15c-4893-9909-90657be7f19f": HorizonUser(
        userId="f6fe6a54-c15c-4893-9909-90657be7f19f",
        name="David",
        userType=HorizonTabs.ENG,
    ),
    "a5d534c9-5426-4387-a298-723c5e09ecab": HorizonUser(
        userId="a5d534c9-5426-4387-a298-723c5e09ecab",
        name="William",
        userType=HorizonTabs.ENG,
    ),
    "74abfc82-4b79-4fb9-b27a-9d2cdff541ce": HorizonUser(
        userId="74abfc82-4b79-4fb9-b27a-9d2cdff541ce",
        name="Mazin",
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
