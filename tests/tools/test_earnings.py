import logging
import os
from unittest import IsolatedAsyncioTestCase

from agent_service.types import PlanRunContext
from agent_service.utils.earnings.earnings_util import (
    get_transcript_partitions_from_db,
    get_transcript_sections_from_partitions,
    insert_transcript_partitions_to_db,
    split_transcript_into_smaller_sections,
)


class TestEarnings(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()

    async def test_transcript_partition_generation(self):
        # Tests the following
        #    split_transcript_into_smaller_sections
        #    group_neighboring_lines

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(curr_dir, "data/earnings_data/google_earnings_call.txt")

        with open(file_path, "r") as file:
            transcript = file.read()
        partition_split = await split_transcript_into_smaller_sections(
            transcript_id=0000, transcript=transcript, context=self.context
        )
        self.assertAlmostEqual(29, len(partition_split.keys()), delta=2)

    async def test_transcript_retrieval(self):
        # Tests the following
        #    get_transcript_partitions_from_db
        #    get_transcript_sections_from_partitions

        transcript_id = "67cb6dc1-c69f-4c02-af69-60ccd0752338"
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(curr_dir, "data/earnings_data/google_earnings_call.txt")

        with open(file_path, "r") as file:
            transcript = file.read()

        partitions = (await get_transcript_partitions_from_db([transcript_id]))[transcript_id]
        transcript_partitions = get_transcript_sections_from_partitions(
            transcript, partitions=partitions
        )
        self.assertEqual(len(partitions), len(transcript_partitions))

    async def test_insert_transcript_partitions(self):
        # Tests the following
        #    insert_transcript_partitions_to_db

        await insert_transcript_partitions_to_db(
            {
                "67cb6dc1-c69f-4c02-af69-60ccd0752338": [
                    [0, 1],
                    [2, 2],
                    [3, 3],
                    [4, 4],
                    [5, 5],
                    [6, 8],
                    [9, 9],
                    [10, 10],
                    [11, 11],
                    [12, 12],
                    [13, 13],
                    [14, 14],
                    [15, 15],
                    [16, 18],
                    [19, 19],
                    [20, 20],
                    [21, 21],
                    [22, 24],
                    [25, 25],
                    [26, 28],
                    [29, 29],
                    [30, 32],
                    [33, 35],
                    [36, 38],
                    [39, 39],
                    [40, 42],
                    [43, 43],
                    [44, 45],
                    [46, 46],
                ]
            }
        )
