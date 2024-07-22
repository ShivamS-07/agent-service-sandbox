import datetime
from typing import List, Optional

from gpt_service_proto_v1.service_grpc import GPTServiceStub
from pydantic.main import BaseModel

from agent_service.GPT.constants import GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import Citation, IOType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable
from agent_service.io_types.text import Text
from agent_service.planner.planner_types import ExecutionPlan, OutputWithID
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.date_utils import timezoneify
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.output_utils.output_construction import PreparedOutput
from agent_service.utils.output_utils.prompts import (
    BASIC_NOTIFICATION_TEMPLATE,
    CUSTOM_NOTIFICATION_TEMPLATE,
    DECIDE_NOTIFICATION_MAIN_PROMPT,
    GENERATE_DIFF_MAIN_PROMPT,
    GENERATE_DIFF_SYS_PROMPT,
    SHORT_DIFF_SUMMARY_MAIN_PROMPT,
    SUMMARY_CUSTOM_NOTIFICATION_TEMPLATE,
    TEXT_OUTPUT_TEMPLATE,
)
from agent_service.utils.output_utils.utils import io_type_to_gpt_input
from agent_service.utils.string_utils import strip_code_backticks

NO_CHANGE_STOCK_LIST_DIFF = "Nothing added or removed from the stock list."


class OutputDiff(BaseModel):
    diff_summary_message: str
    output_id: Optional[str] = None
    should_notify: bool = True
    title: Optional[str] = None


class OutputDiffer:
    def __init__(
        self,
        plan: ExecutionPlan,
        context: PlanRunContext,
        custom_notifications: Optional[str],
        model: str = GPT4_O,
        gpt_service_stub: Optional[GPTServiceStub] = None,
    ):
        self.context = context
        self.plan = plan
        self.custom_notifications = custom_notifications
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

        if self.custom_notifications:
            notification_instructions_str = CUSTOM_NOTIFICATION_TEMPLATE.format(
                custom_notifications=self.custom_notifications
            )
        else:
            notification_instructions_str = BASIC_NOTIFICATION_TEMPLATE

        main_prompt = GENERATE_DIFF_MAIN_PROMPT.format(
            output_schema=OutputDiff.model_json_schema(),
            latest_output=TEXT_OUTPUT_TEMPLATE.format(
                text=(await latest_output.get()).val,
                citations="\n".join(new_citations),
            ).filled_prompt,
            prev_output=f"'{(await prev_output.get()).val}'",
            notification_instructions=notification_instructions_str,
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
        elif (
            isinstance(latest_output, List)
            and isinstance(prev_output, List)
            and (
                (len(latest_output) > 0 and isinstance(latest_output[0], StockID))
                or (len(prev_output) > 0 and isinstance(prev_output[0], StockID))
            )
        ):
            return await self._compute_diff_for_stock_lists(
                curr_stock_list=latest_output, prev_stock_list=prev_output
            )
        elif isinstance(latest_output, StockTable) and isinstance(prev_output, StockTable):
            # try do to a stock list diff first
            stock_list_diff = await self._compute_diff_for_stock_lists(
                curr_stock_list=latest_output.get_stocks(), prev_stock_list=prev_output.get_stocks()
            )
            if stock_list_diff.diff_summary_message is not NO_CHANGE_STOCK_LIST_DIFF:
                return stock_list_diff
            # if no change to listed stocks, do default_diff

        latest_output_str = await io_type_to_gpt_input(latest_output, use_abbreviated_output=False)
        prev_output_str = await io_type_to_gpt_input(prev_output, use_abbreviated_output=False)

        if self.custom_notifications:
            notification_instructions_str = CUSTOM_NOTIFICATION_TEMPLATE.format(
                custom_notifications=self.custom_notifications
            )
        else:
            notification_instructions_str = BASIC_NOTIFICATION_TEMPLATE

        result = await self.llm.do_chat_w_sys_prompt(
            main_prompt=GENERATE_DIFF_MAIN_PROMPT.format(
                output_schema=OutputDiff.model_json_schema(),
                latest_output=latest_output_str,
                prev_output=prev_output_str,
                notification_instructions=notification_instructions_str,
            ),
            sys_prompt=GENERATE_DIFF_SYS_PROMPT,
            output_json=True,
        )
        # TODO handle retries, errors, etc.
        return OutputDiff.model_validate_json(strip_code_backticks(result))

    async def _compute_diff_for_stock_lists(
        self, curr_stock_list: List[StockID], prev_stock_list: List[StockID]
    ) -> OutputDiff:
        if not self.context.diff_info:
            return OutputDiff(diff_summary_message="", should_notify=False)
        curr_stock_set = set(curr_stock_list)
        prev_stock_set = set(prev_stock_list)
        added_stocks = curr_stock_set - prev_stock_set
        removed_stocks = prev_stock_set - curr_stock_set
        final_output = []
        if added_stocks:
            added_output = []
            for stock in added_stocks:
                found = True
                for history_entry in stock.history:
                    if history_entry.task_id is not None:
                        task_id = history_entry.task_id
                        found = False
                        if task_id in self.context.diff_info and stock in self.context.diff_info[
                            task_id
                        ].get("added", []):
                            added_output.append(
                                f"    - {stock.company_name}: {self.context.diff_info[task_id]['added'][stock]}"
                            )
                            break
                if not found:
                    added_output.append(f"    - {stock.company_name}")
            if added_output:
                final_output.append("\n".join(["  - Added stocks"] + added_output))

        if removed_stocks:
            removed_output = []
            for stock in removed_stocks:
                found = False
                for history_entry in stock.history:
                    if history_entry.task_id is not None:
                        task_id = history_entry.task_id
                        if task_id in self.context.diff_info and stock in self.context.diff_info[
                            task_id
                        ].get("removed", []):
                            removed_output.append(
                                f"    - {stock.company_name}: {self.context.diff_info[task_id]['removed'][stock]}"
                            )
                            found = True
                            break
                if not found:
                    removed_output.append(f"    - {stock.company_name}")
            if removed_output:
                final_output.append("\n".join(["  - Removed stocks"] + removed_output))

        if final_output:
            final_str = "\n" + "\n".join(final_output)

        if final_str:
            latest_output_str = await io_type_to_gpt_input(
                curr_stock_list, use_abbreviated_output=False
            )
            prev_output_str = await io_type_to_gpt_input(
                prev_stock_list, use_abbreviated_output=False
            )

            if self.custom_notifications:
                notification_instructions_str = CUSTOM_NOTIFICATION_TEMPLATE.format(
                    custom_notifications=self.custom_notifications
                )
            else:
                notification_instructions_str = BASIC_NOTIFICATION_TEMPLATE

            result = await self.llm.do_chat_w_sys_prompt(
                main_prompt=DECIDE_NOTIFICATION_MAIN_PROMPT.format(
                    output_schema=OutputDiff.model_json_schema(),
                    latest_output=latest_output_str,
                    prev_output=prev_output_str,
                    notification_instructions=notification_instructions_str,
                ),
                sys_prompt=NO_PROMPT,
            )

            if result.lower().startswith("yes"):
                notify = True
            else:
                notify = False

            return OutputDiff(diff_summary_message=final_str, should_notify=notify)
        return OutputDiff(diff_summary_message=NO_CHANGE_STOCK_LIST_DIFF, should_notify=False)

    async def diff_outputs(
        self,
        latest_outputs_with_ids: List[OutputWithID],
        db: BoostedPG,
        prev_outputs: Optional[List[IOType]] = None,
        prev_date: Optional[datetime.datetime] = None,
    ) -> List[OutputDiff]:
        """
        Given a list of the latest outputs for an agent, return a list of output
        diffs for each output. Diffs will be calculated from the last run of the
        agent.
        """

        latest_outputs = [val.output for val in latest_outputs_with_ids]
        async_db = AsyncDB(pg=db)
        if not prev_outputs or not prev_date:
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

        prev_output_values = [
            output.val if isinstance(output, PreparedOutput) else output for output in prev_outputs
        ]
        latest_output_values = [
            output.val if isinstance(output, PreparedOutput) else output
            for output in latest_outputs
        ]

        diffs: List[OutputDiff] = await gather_with_concurrency(
            [
                self._compute_diff_for_io_types(
                    latest_output=latest, prev_output=prev, db=async_db, prev_date=prev_date
                )
                for latest, prev in zip(latest_output_values, prev_output_values)
            ]
        )

        return [
            OutputDiff(
                diff_summary_message=diff.diff_summary_message,
                should_notify=diff.should_notify,
                title=output.output.title if isinstance(output.output, PreparedOutput) else None,
                output_id=output.output_id,
            )
            for output, diff in zip(latest_outputs_with_ids, diffs)
        ]

    async def generate_short_diff_summary(
        self, full_text_diffs: str, notification_criteria: Optional[str]
    ) -> str:
        return await self.llm.do_chat_w_sys_prompt(
            SHORT_DIFF_SUMMARY_MAIN_PROMPT.format(
                diffs=full_text_diffs,
                custom_notifications=SUMMARY_CUSTOM_NOTIFICATION_TEMPLATE.format(
                    notification_criteria=notification_criteria
                ),
            ),
            NO_PROMPT,
            max_tokens=100,
        )
