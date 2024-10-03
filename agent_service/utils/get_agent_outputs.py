import json
import logging
import re
import time
from typing import Any, Optional, cast

from fastapi import HTTPException
from starlette import status

from agent_service.endpoints.models import GetAgentOutputResponse
from agent_service.io_types.text import Text, TextOutput
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.cache_utils import CacheBackend

LOGGER = logging.getLogger(__name__)


async def get_agent_output(
    pg: AsyncDB,
    agent_id: str,
    plan_run_id: Optional[str] = None,
    cache: Optional[CacheBackend] = None,
) -> GetAgentOutputResponse:
    """
    If `plan_run_id` is None, default to get the outputs of the latest run
    """
    t = time.perf_counter()
    if not plan_run_id:
        sql = """SELECT plan_run_id FROM agent.agent_outputs
               WHERE agent_id = %(agent_id)s AND "output" NOTNULL AND is_intermediate = FALSE
               ORDER BY created_at DESC LIMIT 1"""
        rows = await pg.pg.generic_read(sql, params={"agent_id": agent_id})
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No output found for {agent_id=} and {plan_run_id=}",
            )
        plan_run_id = rows[0]["plan_run_id"]
    key = f"{agent_id}-{plan_run_id}"
    if cache:
        cached_val = await cache.get(key)
        if cached_val:
            LOGGER.info(
                f"total time to get output using cache for {agent_id} {time.perf_counter() - t}"
            )
            return cast(GetAgentOutputResponse, cached_val)
    outputs = await pg.get_agent_outputs(agent_id=agent_id, plan_run_id=plan_run_id)
    if not outputs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No output found for {agent_id=} and {plan_run_id=}",
        )

    run_metadata = outputs[0].run_metadata

    newly_updated_outputs = (run_metadata.updated_output_ids or []) if run_metadata else []
    run_summary_short = run_metadata.run_summary_short if run_metadata else None

    run_summary_long: Any = run_metadata.run_summary_long if run_metadata else None
    if isinstance(run_summary_long, Text):
        run_summary_long = await run_summary_long.to_rich_output(pg=pg.pg)
        run_summary_long = cast(TextOutput, run_summary_long)

        # replace summary title with widget anchor
        for output in outputs:
            widget_title = output.output.title
            if widget_title:
                summary_title_dict = {
                    "type": "output_widget",
                    "name": widget_title,
                    "output_id": output.output_id,
                    "plan_run_id": output.plan_run_id,
                }
                summary_title_anchor = "```" + json.dumps(summary_title_dict) + "```"
                run_summary_long.val = re.sub(
                    "- " + widget_title, summary_title_anchor, run_summary_long.val, count=1
                )

        # regex removes all the spaces between any "\n" and "-"" so bullet points
        # can be displayed properly
        run_summary_long.val = re.sub(r"(?<=\n)\s+(?=-)", "", run_summary_long.val)

    final_outputs = [output for output in outputs if not output.is_intermediate]
    result = GetAgentOutputResponse(
        outputs=outputs,
        run_summary_long=run_summary_long,
        run_summary_short=run_summary_short,
        newly_updated_outputs=newly_updated_outputs,
    )
    if final_outputs:
        result = GetAgentOutputResponse(
            outputs=final_outputs,
            run_summary_long=run_summary_long,
            run_summary_short=run_summary_short,
            newly_updated_outputs=newly_updated_outputs,
        )
    if cache:
        await cache.set(key=key, val=result, ttl=60 * 24 * 60 * 60)
        LOGGER.info(f"saved output for {agent_id} to cache")
    LOGGER.info(f"total time to get output for {agent_id} {time.perf_counter() - t}")
    return result
