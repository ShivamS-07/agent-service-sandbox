import copy
import datetime
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional

from gpt_service_proto_v1.service_grpc import GPTServiceStub
from pydantic.main import BaseModel

from agent_service.GPT.constants import GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import Citation, HistoryEntry, IOType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable
from agent_service.io_types.text import Text, TextCitation, TextCitationGroup
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
    TEXT_GENERATE_BASIC_DIFF_PROMPT,
    TEXT_GENERATE_CITE_DIFF_PROMPT,
    TEXT_GENERATE_FINAL_DIFF_PROMPT,
)
from agent_service.utils.output_utils.utils import io_type_to_gpt_input
from agent_service.utils.string_utils import clean_to_json_if_needed

logger = logging.getLogger(__name__)

NO_CHANGE_STOCK_LIST_DIFF = "Nothing added or removed from the stock list."
NO_UPDATE_MESSAGE = "No new updates."


class OutputDiff(BaseModel):
    diff_summary_message: str
    output_id: Optional[str] = None
    citations: Optional[List[TextCitation]] = None
    should_notify: bool = True
    title: Optional[str] = None


def generate_full_diff_summary(diffs: List[OutputDiff]) -> Optional[Text]:
    output_strs = []

    def get_output_len(output_strs: List[str]) -> int:
        return sum([len(S) for S in output_strs])

    all_citations: List[Citation] = []

    for diff in diffs:
        if not diff.diff_summary_message.strip() or diff.diff_summary_message == NO_UPDATE_MESSAGE:
            continue
        output_strs.append("- ")
        if diff.title:
            output_strs.append(diff.title)
            output_strs.append(": ")
        curr_offset = get_output_len(output_strs)
        if diff.citations:
            for citation in diff.citations:
                if citation.citation_text_offset is not None:
                    citation.citation_text_offset += curr_offset
                    all_citations.append(citation)
        output_strs.append(diff.diff_summary_message)
        output_strs.append("\n")
    if not output_strs:
        return None
    text = Text(val="".join(output_strs))
    return text.inject_history_entry(HistoryEntry(citations=all_citations))


def get_citable_changes(result: str) -> Dict[str, List[int]]:
    return {
        triple.split(";")[0]: eval(triple.split(";")[2])
        for triple in result.split("\n")
        if ";" in triple and "yes" in triple.split(";")[1].lower()
    }


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
        self, latest_output: Text, prev_output: Text, prev_run_time: datetime.datetime
    ) -> OutputDiff:
        old_citation_texts = [
            citation.source_text
            for citation in prev_output.get_all_citations()
            if isinstance(citation, TextCitation)
        ]
        for old_text in old_citation_texts:
            old_text.reset_id()
        old_citation_text_set = set(old_citation_texts)
        recent_citations = [
            citation
            for citation in latest_output.get_all_citations()
            if isinstance(citation, TextCitation)
            and citation.source_text not in old_citation_text_set
            and citation.source_text.timestamp
            and timezoneify(citation.source_text.timestamp)
            > timezoneify(prev_run_time) - datetime.timedelta(days=1)
        ]  # add an extra day here in case there's lag between publication and availability in db

        if not recent_citations:
            return OutputDiff(diff_summary_message=NO_UPDATE_MESSAGE, should_notify=False)

        citation_group = TextCitationGroup(val=recent_citations)
        citation_str = await citation_group.convert_to_str()

        if self.custom_notifications:
            notification_instructions_str = CUSTOM_NOTIFICATION_TEMPLATE.format(
                custom_notifications=self.custom_notifications
            )
        else:
            notification_instructions_str = BASIC_NOTIFICATION_TEMPLATE

        latest_output_text = (await latest_output.get()).val
        prev_output_text = (await prev_output.get()).val

        basic_prompt = TEXT_GENERATE_BASIC_DIFF_PROMPT.format(
            latest_output=latest_output_text, prev_output=prev_output_text
        )
        result = await self.llm.do_chat_w_sys_prompt(
            main_prompt=basic_prompt,
            sys_prompt=NO_PROMPT,
        )

        try:
            changes = [
                pair.split(":")[0]
                for pair in result.split("\n")
                if "no" in pair.split(":")[-1].lower()
            ]
        except IndexError:
            result = await self.llm.do_chat_w_sys_prompt(
                main_prompt=basic_prompt,
                sys_prompt=NO_PROMPT,
            )
            changes = [
                pair.split(":")[0]
                for pair in result.split("\n")
                if "no" in pair.split(":")[-1].lower()
            ]

        if not changes:
            return OutputDiff(diff_summary_message=NO_UPDATE_MESSAGE, should_notify=False)

        cite_prompt = TEXT_GENERATE_CITE_DIFF_PROMPT.format(
            output=latest_output_text, citations=citation_str, topics="\n".join(changes)
        )
        result = await self.llm.do_chat_w_sys_prompt(
            main_prompt=cite_prompt,
            sys_prompt=NO_PROMPT,
        )

        try:
            citable_changes = get_citable_changes(result)
        except (IndexError, SyntaxError):
            result = await self.llm.do_chat_w_sys_prompt(
                main_prompt=cite_prompt,
                sys_prompt=NO_PROMPT,
            )
            citable_changes = get_citable_changes(result)

        if not citable_changes:
            return OutputDiff(diff_summary_message=NO_UPDATE_MESSAGE, should_notify=False)

        final_prompt = TEXT_GENERATE_FINAL_DIFF_PROMPT.format(
            changes="\n".join(citable_changes.keys()),
            latest_output=(await latest_output.get()).val,
            prev_output=(await prev_output.get()).val,
            notification_instructions=notification_instructions_str,
        )

        result = await self.llm.do_chat_w_sys_prompt(
            main_prompt=final_prompt,
            sys_prompt=NO_PROMPT,
        )

        diff_sents, notify = result.split("!!!")

        diff_text_bits = []
        citations = []
        for topic_and_sentence in diff_sents.split("\n"):
            if not topic_and_sentence.strip() or topic_and_sentence.count(": ") != 1:
                continue
            topic, sentence = topic_and_sentence.split(": ")
            if topic in citable_changes:
                diff_text_bits.append("\n    - ")
                diff_text_bits.append(sentence)
                offset = sum([len(S) for S in diff_text_bits])
                for citation_num in citable_changes[topic]:
                    citation = citation_group.convert_citation_num_to_citation(citation_num)
                    if citation:
                        citation = copy.deepcopy(citation)
                        citation.citation_text_offset = offset - 1
                        citations.append(citation)

        diff_text = "".join(diff_text_bits)

        return OutputDiff(
            diff_summary_message=diff_text,
            should_notify="yes" in notify.lower(),
            citations=citations,
        )

    async def _compute_diff_for_io_types(
        self, latest_output: IOType, prev_output: IOType, prev_run_time: datetime.datetime
    ) -> OutputDiff:
        if isinstance(latest_output, Text) and isinstance(prev_output, Text):
            return await self._compute_diff_for_texts(
                latest_output=latest_output, prev_output=prev_output, prev_run_time=prev_run_time
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
                latest_output=latest_output_str,
                special_instructions="",
                no_update_message=NO_UPDATE_MESSAGE,
                prev_output=prev_output_str,
                notification_instructions=notification_instructions_str,
            ),
            sys_prompt=GENERATE_DIFF_SYS_PROMPT,
            output_json=True,
        )
        # TODO handle retries, errors, etc.
        result_json = json.loads(clean_to_json_if_needed(result))
        return OutputDiff(
            diff_summary_message=result_json["diff_summary_message"],
            should_notify=result_json["should_notify"],
        )

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
                            found = True
                            break
                if not found:
                    added_output.append(f"    - {stock.company_name}")
            if added_output:
                final_output.append("\n".join(["  - Added stocks"] + added_output))

        final_str = None
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
        prev_run_time: Optional[datetime.datetime] = None,
    ) -> List[OutputDiff]:
        """
        Given a list of the latest outputs for an agent, return a list of output
        diffs for each output. Diffs will be calculated from the last run of the
        agent.
        """

        latest_outputs = [val.output for val in latest_outputs_with_ids]
        async_db = AsyncDB(pg=db)
        if not prev_outputs or not prev_run_time:
            prev_outputs_and_time = await async_db.get_prev_outputs_for_agent_plan(
                agent_id=self.context.agent_id,
                plan_id=self.context.plan_id,
                latest_plan_run_id=self.context.plan_run_id,
            )
            if prev_outputs_and_time is None:
                # If this is the first run of the plan, or the previous output is
                # misisng for some reason.
                logger.info("No previous output found, won't notify.")
                return [
                    OutputDiff(
                        diff_summary_message="",
                        should_notify=False,
                    )
                ]

            prev_outputs, prev_run_time = prev_outputs_and_time
        if len(prev_outputs) != len(latest_outputs):
            # Assuming the plan is the same, the number of outputs should ideally be
            # the same also. If not, we can't easily compare.
            logger.info(
                (
                    f"Got {len(prev_outputs)} prev outputs and "
                    f"{len(latest_outputs)} latest outputs, cannot diff."
                )
            )
            return [
                OutputDiff(
                    diff_summary_message="",
                    should_notify=False,
                )
            ]

        output_pairs = defaultdict(list)

        for output_lists in [latest_outputs, prev_outputs]:
            for output in output_lists:
                if isinstance(output, PreparedOutput):  # only allow PreparedOutput with titles
                    output_pairs[output.title].append(output.val)

        for title in list(output_pairs.keys()):  # remove anything that isn't pairs
            if len(output_pairs[title]) != 2:
                del output_pairs[title]

        diffs: List[OutputDiff] = await gather_with_concurrency(
            [
                self._compute_diff_for_io_types(
                    latest_output=latest, prev_output=prev, prev_run_time=prev_run_time
                )
                for latest, prev in output_pairs.values()
            ]
        )

        return [
            OutputDiff(
                diff_summary_message=diff.diff_summary_message,
                should_notify=diff.should_notify,
                citations=diff.citations,
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
