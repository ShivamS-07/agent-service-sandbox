import logging
from typing import Dict, List, Optional, Set, Tuple

from agent_service.GPT.constants import GPT4_O_MINI
from agent_service.GPT.requests import GPT
from agent_service.types import PlanRunContext
from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.earnings.prompts import (
    TRANSCRIPT_PARTITION_MAIN_PROMPT,
    TRANSCRIPT_PARTITION_SYS_PROMPT,
)
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context

logger = logging.getLogger(__name__)


def group_neighboring_lines(line_nums: List[int]) -> List[List[int]]:
    if not line_nums:
        return []

    result = []
    current_group = [line_nums[0]]

    for i in range(1, len(line_nums)):
        if line_nums[i] == line_nums[i - 1] + 1:
            current_group.append(line_nums[i])
        else:
            result.append(current_group)
            current_group = [line_nums[i]]

    result.append(current_group)  # Append the last group
    return result


async def get_transcript_partitions_from_db(
    transcript_ids: List[str],
) -> Dict[str, List[List[int]]]:
    sql = """
    SELECT transcript_id, partitions
    FROM company_earnings.earning_transcripts_partitions
    WHERE transcript_id IN %(transcript_ids)s
    """

    ch = Clickhouse()
    res = await ch.generic_read(
        sql,
        params={
            "transcript_ids": transcript_ids,
        },
    )
    return {str(row["transcript_id"]): row["partitions"] for row in res}


async def insert_transcript_partitions_to_db(
    partition_data: Dict[str, List[Tuple[int, int]]],
) -> None:
    records_to_upload_to_db = []
    for transcript_id, partitions in partition_data.items():
        records_to_upload_to_db.append({"transcript_id": transcript_id, "partitions": partitions})
    ch = Clickhouse()
    await ch.multi_row_insert(
        table_name="company_earnings.earning_transcripts_partitions", rows=records_to_upload_to_db
    )


def get_transcript_sections_from_partitions(
    transcript: str, partitions: List[List[int]]
) -> Dict[Tuple[int, int], str]:
    lines = transcript.split("\n")
    partitions_dict: Dict[Tuple[int, int], str] = {}
    for partition in partitions:
        starting_line = min(partition)
        ending_line = max(partition)
        if starting_line == ending_line:
            partitions_dict[(starting_line, ending_line)] = lines[starting_line]
        else:
            partitions_dict[(starting_line, ending_line)] = "\n".join(
                lines[starting_line:ending_line]
            )
    return partitions_dict


async def split_transcript_into_smaller_sections(
    transcript_id: str, transcript: str, context: PlanRunContext, llm: Optional[GPT] = None
) -> Dict[Tuple[int, int], str]:
    if not llm:
        gpt_context = create_gpt_context(
            GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
        )
        llm = GPT(model=GPT4_O_MINI, context=gpt_context)

    lines = transcript.split("\n")
    numbered_transcript = "\n".join([f"{i + 1}. {line}" for i, line in enumerate(lines)])

    res = await llm.do_chat_w_sys_prompt(
        main_prompt=TRANSCRIPT_PARTITION_MAIN_PROMPT.format(transcript_text=numbered_transcript),
        sys_prompt=TRANSCRIPT_PARTITION_SYS_PROMPT.format(),
    )
    partitions_dict: Dict[Tuple[int, int], str] = {}
    saved_line_numbers: Set[int] = set()

    prev_ending_line = 0
    for raw_partition in res.split("\n"):
        try:
            partition = list(map(int, raw_partition.replace(" ", "").strip().split(",")))

            # Subtract 1 to account for the offset we added then ensure
            # the starting line is not overlapping with the older partition
            starting_line = max(min(partition) - 1, prev_ending_line)

            # GPT has a tendency to go over the actual line numbers
            ending_line = min(max(partition) - 1, len(lines) - 1)

            if starting_line == ending_line:
                partitions_dict[(starting_line, ending_line)] = lines[starting_line]
                saved_line_numbers.update([starting_line])
            else:
                prev_ending_line = ending_line
                partitions_dict[(starting_line, ending_line)] = "\n".join(
                    lines[starting_line:ending_line]
                )
                saved_line_numbers.update(range(starting_line, ending_line + 1))
            prev_ending_line = ending_line
        except (IndexError, ValueError):
            logger.warning(
                f"Failed to properly partition transcript {transcript_id}, "
                f"gpt output: {raw_partition}"
            )
            continue

    missed_line_numbers = set(range(0, len(lines))) - saved_line_numbers

    # If any lines were missed somehow, we add them back in
    if missed_line_numbers:
        # See if any lines missed were neighboring line numbers
        grouped_lines = group_neighboring_lines(list(missed_line_numbers))
        for group in grouped_lines:
            # If we just missed a single short line, then it probably wasn't very important
            if (len(group) == 1) and (len(lines[group[0]].split(" "))) < 10:
                continue

            try:
                if len(group) == 1:
                    partitions_dict[(group[0], group[0])] = "\n".join([lines[i] for i in group])
                else:
                    partitions_dict[(group[0], group[-1])] = "\n".join([lines[i] for i in group])
            except IndexError:
                logger.warning(
                    f"Failed to get index for {group} in transcript {transcript_id}, "
                    f"which had {len(lines)} lines"
                )
                continue

    return partitions_dict
