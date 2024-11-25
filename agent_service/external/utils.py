from typing import Optional

import aiohttp

HTTP_SESSION: Optional[aiohttp.ClientSession] = None


def get_http_session() -> aiohttp.ClientSession:
    global HTTP_SESSION
    if HTTP_SESSION is None:
        HTTP_SESSION = aiohttp.ClientSession()
    return HTTP_SESSION
