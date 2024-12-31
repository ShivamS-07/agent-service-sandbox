import asyncio
import datetime
import time
from typing import Any, NamedTuple

from tqdm.asyncio import tqdm_asyncio

from agent_service.utils.redis_queue import publish_agent_event


# TODO Importing PlanRunContext causes a circular import issue. The import tree is pretty messy,
# we will fix this later.
class ProgressBarArgs(NamedTuple):
    context: Any  # PlanRunContext
    desc: str


class FrontendProgressBar(tqdm_asyncio):
    def __init__(
        self,
        *args: Any,
        context: Any,
        log_id: str,
        created_at: datetime.datetime,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.last_published_time = -1.0
        self.context = context
        self.log_id = log_id
        self.created_at = created_at

    @classmethod
    async def all_done(
        cls, context: Any, log_id: str, desc: str, created_at: datetime.datetime
    ) -> None:
        current_percentage = 1.0
        from agent_service.endpoints.models import AgentEvent, TaskLogProgressBarEvent

        agent_event = AgentEvent(
            agent_id=context.agent_id,
            event=TaskLogProgressBarEvent(
                percentage=current_percentage,
                log_id=log_id,
                created_at=created_at,
                log_message=desc,
                task_id=context.task_id,
            ),
        )
        await publish_agent_event(
            context.agent_id,
            agent_event.model_dump_json(),
        )
        from agent_service.tools.tool_log import tool_log

        await tool_log(log=desc, context=context, log_id=log_id, percentage=current_percentage)

    # pyanalyze: ignore[missing_await]
    def update(self, n: int = 1) -> None:
        super().update(n)
        from agent_service.endpoints.models import AgentEvent, TaskLogProgressBarEvent

        current_time = time.time()
        if self.total:
            current_percentage = round((self.n / self.total), 2)

            if (current_time - self.last_published_time >= 5.0) or (
                current_percentage >= 1.0 and self.n == self.total
            ):
                self.last_published_time = current_time
                agent_event = AgentEvent(
                    agent_id=self.context.agent_id,
                    event=TaskLogProgressBarEvent(
                        percentage=current_percentage,
                        log_id=self.log_id,
                        created_at=self.created_at,
                        log_message=self.desc,
                        task_id=self.context.task_id,
                    ),
                )
                asyncio.create_task(  # static analysis: ignore
                    publish_agent_event(
                        self.context.agent_id,
                        agent_event.model_dump_json(),
                    )
                )
                from agent_service.tools.tool_log import tool_log

                asyncio.create_task(  # static analysis: ignore
                    tool_log(
                        log=self.desc,
                        context=self.context,
                        log_id=self.log_id,
                        percentage=current_percentage,
                    )
                )
