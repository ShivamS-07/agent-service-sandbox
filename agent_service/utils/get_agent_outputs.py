import asyncio
import json
import logging
import re
import time
from typing import Any, Optional, cast

from fastapi import HTTPException
from starlette import status

from agent_service.endpoints.models import (
    GetAgentOutputResponse,
)
from agent_service.io_types.table import TableOutput, TableTransformation
from agent_service.io_types.text import Text, TextOutput
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.cache_utils import RedisCacheBackend

LOGGER = logging.getLogger(__name__)


async def get_agent_output(
    pg: AsyncDB,
    agent_id: str,
    plan_run_id: Optional[str] = None,
    cache: Optional[RedisCacheBackend] = None,
) -> GetAgentOutputResponse:
    """
    If `plan_run_id` is None, default to get the outputs of the latest run
    """
    t = time.perf_counter()

    outputs, quick_thoughts, output_transformations = await asyncio.gather(
        pg.get_agent_outputs(agent_id=agent_id, plan_run_id=plan_run_id, cache=cache),
        pg.get_latest_quick_thought_for_agent(agent_id=agent_id),
        pg.get_output_transformations(agent_id=agent_id, plan_run_id=plan_run_id),
    )

    if plan_run_id:
        # Don't include quick thoughts if they're asking for output from a
        # specific run. ONLY for the latest output.
        quick_thoughts = None
    if not outputs and not quick_thoughts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No output found for {agent_id=} and {plan_run_id=}",
        )

    run_summary_long: Any = None
    run_summary_short = None
    newly_updated_outputs: list[str] = []
    if outputs:
        run_metadata = outputs[0].run_metadata

        newly_updated_outputs = (run_metadata.updated_output_ids or []) if run_metadata else []
        run_summary_short = run_metadata.run_summary_short if run_metadata else None

        run_summary_long = run_metadata.run_summary_long if run_metadata else None
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

            run_summary_long.val = run_summary_long.val.replace("\n  ", "\n")

        for output in outputs:
            transformation = output_transformations.get(output.task_id)
            if transformation:
                LOGGER.info(f"Applying transformation to {output.output_id}")
                output.transformation_id = transformation["transformation_id"]
                output.is_transformation_local = transformation["is_transformation_local"]
                if isinstance(output.output, TableOutput):
                    output.output.transform(
                        TableTransformation.model_validate(transformation["transformation"])
                    )

    LOGGER.info(f"total time to get output for {agent_id} is {(time.perf_counter() - t):.2f}s")

    final_outputs = [output for output in outputs if not output.is_intermediate]
    return GetAgentOutputResponse(
        outputs=final_outputs or outputs,
        run_summary_long=run_summary_long,
        run_summary_short=run_summary_short,
        newly_updated_outputs=newly_updated_outputs,
        quick_thoughts=quick_thoughts,
    )
