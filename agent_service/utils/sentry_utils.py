import logging
import os
from typing import Any, Optional

import sentry_sdk
from gbi_common_py_utils.utils.environment import get_environment_tag

EVENTS = []  # type: ignore
SERVICE_SENTRY_DSN = "https://a571d1cbf1f90b72451e3328c17a9514@sentry.boosted.ai/26"

logger = logging.getLogger(__name__)


def init_sentry(
    transport: Any = None,
    debug: bool = False,
    disable_sentry: bool = False,
) -> None:
    """
    Set transport = DebugTransport() to not log tests to the UI.
    """

    if os.getenv("NO_SENTRY"):
        logger.info("Sentry disabled with env var")
        return

    tag = get_environment_tag()
    traces_sample_rate = 0.0

    if tag in ["ALPHA", "DEV", "STAGING"]:
        traces_sample_rate = 1.0
    else:
        # disable locally
        logger.info(f"Skipping sentry for - env: {tag} ...")
        return

    logger.info(f"Initializing Sentry - env: {tag} ...")
    dsn: Optional[str] = SERVICE_SENTRY_DSN
    if disable_sentry:
        # Recommended from: https://github.com/getsentry/sentry-python/issues/1668
        logger.warning("Disabling sentry. This is an ERROR if you are not running locally.")
        dsn = None

    global EVENTS
    sentry_sdk.init(
        debug=debug,
        transport=transport,
        dsn=dsn,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=traces_sample_rate,
        environment=tag,
    )
