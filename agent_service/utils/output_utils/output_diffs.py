import copy
import datetime
import json
import logging
from collections import defaultdict
from itertools import islice
from typing import Dict, List, Optional, Tuple

from gpt_service_proto_v1.service_grpc import GPTServiceStub
from pydantic.main import BaseModel

from agent_service.GPT.constants import GPT4_O, NO_PROMPT
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.io_type_utils import Citation, HistoryEntry, IOType
from agent_service.io_types.stock import StockID
from agent_service.io_types.table import StockTable
from agent_service.io_types.text import Text, TextCitation, TextCitationGroup
from agent_service.planner.planner_types import ExecutionPlan, OutputWithID
from agent_service.types import PlanRunContext
from agent_service.utils.async_db import AsyncDB
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.constants import MAX_CITABLE_CHANGES_PER_WIDGET
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

NO_CHANGE_STOCK_LIST_DIFF = "No significant changes to the list of stocks."
NO_UPDATE_MESSAGE = "No new updates."

ADDED_HEADER = "**Newly Added Stocks:**\n"
REMOVED_HEADER = "**Removed Stocks:**\n"


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
    ) -> Tuple[OutputDiff, List[Tuple[str, List[TextCitation]]]]:
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
            return OutputDiff(diff_summary_message=NO_UPDATE_MESSAGE, should_notify=False), []

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
            return OutputDiff(diff_summary_message=NO_UPDATE_MESSAGE, should_notify=False), []

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
            return OutputDiff(diff_summary_message=NO_UPDATE_MESSAGE, should_notify=False), []
        elif len(citable_changes) > MAX_CITABLE_CHANGES_PER_WIDGET:
            citable_changes = dict(islice(citable_changes.items(), MAX_CITABLE_CHANGES_PER_WIDGET))

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
        all_citations = []
        # save sentence/citation pairs for use in constructing larger outputs
        citation_sentence_pairs = []
        for topic_and_sentence in diff_sents.split("\n"):
            if not topic_and_sentence.strip() or topic_and_sentence.count(": ") != 1:
                continue
            topic, sentence = topic_and_sentence.split(": ")
            if topic in citable_changes and citable_changes[topic]:
                diff_text_bits.append("\n  - ")
                diff_text_bits.append(sentence)
                offset = sum([len(S) for S in diff_text_bits])
                sent_citations = []
                for citation_num in citable_changes[topic]:
                    citation = citation_group.convert_citation_num_to_citation(citation_num)
                    if citation:
                        citation = copy.deepcopy(citation)
                        citation.citation_text_offset = offset - 1
                        sent_citations.append(citation)
                all_citations.extend(sent_citations)
                citation_sentence_pairs.append((sentence, sent_citations))

        diff_text = "".join(diff_text_bits)

        return (
            OutputDiff(
                diff_summary_message=diff_text,
                should_notify="yes" in notify.lower(),
                citations=all_citations,
            ),
            citation_sentence_pairs,
        )

    async def _compute_diff_for_io_types(
        self, latest_output: IOType, prev_output: IOType, prev_run_time: datetime.datetime
    ) -> OutputDiff:
        if isinstance(latest_output, Text) and isinstance(prev_output, Text):
            return (
                await self._compute_diff_for_texts(
                    latest_output=latest_output,
                    prev_output=prev_output,
                    prev_run_time=prev_run_time,
                )
            )[0]
        elif (
            isinstance(latest_output, List)
            and isinstance(prev_output, List)
            and (
                (len(latest_output) > 0 and isinstance(latest_output[0], StockID))
                or (len(prev_output) > 0 and isinstance(prev_output[0], StockID))
            )
        ):
            return await self._compute_diff_for_stock_lists(
                curr_stock_list=latest_output,
                prev_stock_list=prev_output,
                prev_run_time=prev_run_time,
            )
        elif isinstance(latest_output, StockTable) and isinstance(prev_output, StockTable):
            # try do to a stock list diff first
            stock_list_diff = await self._compute_diff_for_stock_lists(
                curr_stock_list=latest_output.get_stocks(),
                prev_stock_list=prev_output.get_stocks(),
                prev_run_time=prev_run_time,
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

        latest_output_str, prev_output_str = GPTTokenizer(GPT4_O).do_multi_truncation_if_needed(
            [latest_output_str, prev_output_str],
            [
                GENERATE_DIFF_MAIN_PROMPT.template,
                GENERATE_DIFF_SYS_PROMPT.filled_prompt,
                NO_UPDATE_MESSAGE,
                notification_instructions_str,
            ],
        )

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
        self,
        curr_stock_list: List[StockID],
        prev_stock_list: List[StockID],
        prev_run_time: datetime.datetime,
    ) -> OutputDiff:
        all_citations: List[TextCitation] = []

        curr_stock_set = set(curr_stock_list)
        prev_stock_set = set(prev_stock_list)
        added_stocks = curr_stock_set - prev_stock_set
        removed_stocks = prev_stock_set - curr_stock_set
        add_remove_output = ["\n"]
        if added_stocks and self.context.diff_info:
            add_remove_output.append(ADDED_HEADER)
            added_stocks_count = 1
            for stock in added_stocks:
                found = True
                for history_entry in stock.history:
                    if history_entry.task_id is not None:
                        task_id = history_entry.task_id
                        found = False
                        if task_id in self.context.diff_info and stock in self.context.diff_info[
                            task_id
                        ].get("added", []):
                            add_remove_output.append(
                                f"  {added_stocks_count}. **{stock.company_name}**  "
                            )
                            if isinstance(self.context.diff_info[task_id]["added"][stock], str):
                                add_remove_output.append(
                                    self.context.diff_info[task_id]["added"][stock]
                                )
                            else:
                                explanation, citations = self.context.diff_info[task_id]["added"][
                                    stock
                                ]
                                add_remove_output.append(explanation)
                                for citation in citations:
                                    citation.citation_text_offset = (
                                        len("".join(add_remove_output)) - 1
                                    )
                                all_citations.extend(citations[:5])
                            found = True
                            break
                if not found:
                    add_remove_output.append(f"  {added_stocks_count}. **{stock.company_name}**")
                add_remove_output.append("\n")
                added_stocks_count += 1

        final_str = None
        if removed_stocks and self.context.diff_info:
            removed_stocks_count = 1
            add_remove_output.append(REMOVED_HEADER)
            for stock in removed_stocks:
                found = False
                for history_entry in stock.history:
                    if history_entry.task_id is not None:
                        task_id = history_entry.task_id
                        if task_id in self.context.diff_info and stock in self.context.diff_info[
                            task_id
                        ].get("removed", []):
                            add_remove_output.append(
                                f"  {removed_stocks_count}. **{stock.company_name}**:  "
                            )
                            if isinstance(self.context.diff_info[task_id]["removed"][stock], str):
                                add_remove_output.append(
                                    self.context.diff_info[task_id]["removed"][stock]
                                )
                            else:
                                explanation, citations = self.context.diff_info[task_id]["removed"][
                                    stock
                                ]
                                add_remove_output.append(explanation)
                                for citation in citations:
                                    citation.citation_text_offset = (
                                        len("".join(add_remove_output)) - 1
                                    )
                                all_citations.extend(citations[:5])
                            found = True
                            break
                if not found:
                    add_remove_output.append(f"  {removed_stocks_count}. **{stock.company_name}**")
                add_remove_output.append("\n")
                removed_stocks_count += 1

        if add_remove_output:
            final_str = "".join(add_remove_output)

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

            latest_output_str, prev_output_str = GPTTokenizer(GPT4_O).do_multi_truncation_if_needed(
                [latest_output_str, prev_output_str],
                [DECIDE_NOTIFICATION_MAIN_PROMPT.template, notification_instructions_str],
            )

            result = await self.llm.do_chat_w_sys_prompt(
                main_prompt=DECIDE_NOTIFICATION_MAIN_PROMPT.format(
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

            curr_offset = len(final_str)
        else:
            final_str = "\n"
            notify = False
            curr_offset = 1

        prev_stock_lookup = {stock: stock for stock in prev_stock_set}

        if len(curr_stock_set) > len(added_stocks):  # there are some shared stocks
            change_output = ["**Modified Stocks:**\n"]
            curr_offset += len(change_output[0])
            modified_stocks_count = 1
            for stock in curr_stock_list:
                if stock in added_stocks:
                    continue
                temp_change_output = [f"  {modified_stocks_count}. **{stock.company_name}**\n"]
                modified_stocks_count += 1
                temp_offset = curr_offset + len(temp_change_output[0])

                for new_history_entry in stock.history:
                    if (
                        new_history_entry.task_id is not None
                        and self.context.diff_info
                        and new_history_entry.task_id in self.context.diff_info
                        and stock
                        in self.context.diff_info[new_history_entry.task_id].get("modified", [])
                    ):
                        # Filter rank change
                        if isinstance(
                            self.context.diff_info[new_history_entry.task_id]["modified"][stock],
                            str,
                        ):
                            sentence = self.context.diff_info[new_history_entry.task_id][
                                "modified"
                            ][stock]
                            citations = []
                        else:
                            sentence, citations = self.context.diff_info[new_history_entry.task_id][
                                "modified"
                            ][stock]

                        bullet = f"      - {sentence}\n"
                        temp_offset += len(bullet)
                        if citations:
                            for citation in citations:
                                citation.citation_text_offset = temp_offset - 2
                            all_citations.extend(citations[:5])
                        temp_change_output.append(bullet)

                    # we are only interested in the output of per stock summary
                    # recommendation and filter output NOT properly updated (yet)
                    elif (
                        isinstance(new_history_entry.explanation, str)
                        and new_history_entry.citations
                        and new_history_entry.title
                        and "Recommendation" not in new_history_entry.title
                        and "Connection to" not in new_history_entry.title
                    ):
                        for old_history_entry in prev_stock_lookup[stock].history:
                            if (
                                old_history_entry.title == new_history_entry.title
                                and isinstance(old_history_entry.explanation, str)
                                and old_history_entry.explanation != new_history_entry.explanation
                            ):
                                new_text = Text(val=new_history_entry.explanation)
                                new_text = new_text.inject_history_entry(
                                    HistoryEntry(citations=new_history_entry.citations)
                                )
                                old_text = Text(val=old_history_entry.explanation)
                                old_text = old_text.inject_history_entry(
                                    HistoryEntry(citations=old_history_entry.citations)
                                )
                                (
                                    output_diff,
                                    sentence_citation_pairs,
                                ) = await self._compute_diff_for_texts(
                                    latest_output=new_text,
                                    prev_output=old_text,
                                    prev_run_time=prev_run_time,
                                )
                                notify |= output_diff.should_notify
                                for sentence, citations in sentence_citation_pairs:
                                    bullet = f"      - {sentence}\n"
                                    temp_offset += len(bullet)
                                    for citation in citations:
                                        citation.citation_text_offset = temp_offset - 2
                                    all_citations.extend(citations[:5])
                                    temp_change_output.append(bullet)

                if len(temp_change_output) > 1:
                    curr_offset = temp_offset
                    change_output.extend(temp_change_output)

            if len(change_output) > 1:
                final_str += "".join(change_output)

        if final_str:
            return OutputDiff(
                diff_summary_message=final_str.rstrip(),
                citations=all_citations,
                should_notify=notify,
            )
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
                cutoff_dt=self.context.as_of_date,
            )
            if prev_outputs_and_time is None:
                # If this is the first run of the plan, or the previous output is
                # missing for some reason.
                logger.info("No previous output found, won't notify.")
                # TODO in the case where this is not the first run of the plan
                # (im sure there is some way to tell)
                # BUT the prev output is missing for some reason (bug/error) etc,
                # it actually would be most prudent to send a notification equivalent to
                # a report for a new agent
                # so client doesnt miss anything...

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
