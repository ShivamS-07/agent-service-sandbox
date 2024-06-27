import datetime
from typing import List, Optional

from gpt_service_proto_v1.service_grpc import GPTServiceStub
from pydantic.main import BaseModel

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import Citation, IOType
from agent_service.io_types.text import Text
from agent_service.planner.planner_types import ExecutionPlan
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.date_utils import timezoneify
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.output_utils.output_construction import PreparedOutput
from agent_service.utils.output_utils.prompts import (
    GENERATE_DIFF_MAIN_PROMPT,
    GENERATE_DIFF_SYS_PROMPT,
    TEXT_OUTPUT_TEMPLATE,
)
from agent_service.utils.output_utils.utils import io_type_to_gpt_input
from agent_service.utils.string_utils import strip_code_backticks


class OutputDiff(BaseModel):
    diff_summary_message: str
    should_notify: bool = True


class OutputDiffer:
    def __init__(
        self,
        plan: ExecutionPlan,
        context: PlanRunContext,
        model: str = GPT4_O,
        gpt_service_stub: Optional[GPTServiceStub] = None,
    ):
        self.context = context
        self.plan = plan
        gpt_context = create_gpt_context(
            GptJobType.AGENT_CHATBOT, self.context.agent_id, GptJobIdType.AGENT_ID
        )
        self.llm = GPT(gpt_context, model, gpt_service_stub=gpt_service_stub)

    async def _compute_diff_for_texts(
        self, latest_output: Text, prev_output: Text, db: AsyncDB, prev_date: datetime.datetime
    ) -> OutputDiff:
        latest_citations = await Citation.resolve_all_citations(
            citations=latest_output.get_all_citations(), db=db.pg
        )
        new_citations = [
            f"*{citation.name}*\n{citation.summary}"
            for citation in latest_citations
            # Ideally both of these should have timezones, but if not just assume they're UTC
            if citation.published_at and timezoneify(citation.published_at) > timezoneify(prev_date)
        ]
        if not new_citations:
            return OutputDiff(diff_summary_message="No new updates.", should_notify=False)

        main_prompt = GENERATE_DIFF_MAIN_PROMPT.format(
            output_schema=OutputDiff.model_json_schema(),
            latest_output=TEXT_OUTPUT_TEMPLATE.format(
                text=(await latest_output.get()).val,
                citations="\n".join(new_citations),
            ).filled_prompt,
            prev_output=f"'{(await prev_output.get()).val}'",
        )
        result = await self.llm.do_chat_w_sys_prompt(
            main_prompt=main_prompt,
            sys_prompt=GENERATE_DIFF_SYS_PROMPT,
            output_json=True,
        )
        return OutputDiff.model_validate_json(strip_code_backticks(result))

    async def _compute_diff_for_io_types(
        self, latest_output: IOType, prev_output: IOType, db: AsyncDB, prev_date: datetime.datetime
    ) -> OutputDiff:
        if isinstance(latest_output, Text) and isinstance(prev_output, Text):
            return await self._compute_diff_for_texts(
                latest_output=latest_output, prev_output=prev_output, db=db, prev_date=prev_date
            )
        latest_output_str = await io_type_to_gpt_input(latest_output, use_abbreviated_output=False)
        prev_output_str = await io_type_to_gpt_input(prev_output, use_abbreviated_output=False)
        result = await self.llm.do_chat_w_sys_prompt(
            main_prompt=GENERATE_DIFF_MAIN_PROMPT.format(
                output_schema=OutputDiff.model_json_schema(),
                latest_output=latest_output_str,
                prev_output=prev_output_str,
            ),
            sys_prompt=GENERATE_DIFF_SYS_PROMPT,
            output_json=True,
        )
        # TODO handle retries, errors, etc.
        return OutputDiff.model_validate_json(strip_code_backticks(result))

    async def diff_outputs(
        self,
        latest_outputs: List[IOType],
        db: BoostedPG,
    ) -> List[OutputDiff]:
        """
        Given a list of the latest outputs for an agent, return a list of output
        diffs for each output. Diffs will be calculated from the last run of the
        agent.
        """

        async_db = AsyncDB(pg=db)
        prev_outputs_and_date = await async_db.get_prev_outputs_for_agent_plan(
            agent_id=self.context.agent_id,
            plan_id=self.context.plan_id,
            latest_plan_run_id=self.context.plan_run_id,
        )
        # If this is the first run of the plan, or the previous output is
        # misisng for some reason, notify just to be safe.
        if prev_outputs_and_date is None:
            return [
                OutputDiff(
                    diff_summary_message="Agent completed with new outputs!",
                    should_notify=True,
                )
            ]

        # Assuming the plan is the same, the number of outputs should ideally be
        # the same also. If not, we can't easily compare, so notify just to be
        # safe.
        prev_outputs, prev_date = prev_outputs_and_date
        if len(prev_outputs) != len(latest_outputs):
            return [
                OutputDiff(
                    diff_summary_message="Agent completed with new outputs!",
                    should_notify=True,
                )
            ]

        prev_outputs = [
            output.val if isinstance(output, PreparedOutput) else output for output in prev_outputs
        ]
        latest_outputs = [
            output.val if isinstance(output, PreparedOutput) else output
            for output in latest_outputs
        ]

        return await gather_with_concurrency(
            [
                self._compute_diff_for_io_types(
                    latest_output=latest, prev_output=prev, db=async_db, prev_date=prev_date
                )
                for latest, prev in zip(latest_outputs, prev_outputs)
            ]
        )
