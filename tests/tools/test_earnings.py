import logging
import os
from unittest import IsolatedAsyncioTestCase
from datetime import datetime
from agent_service.io_types.dates import DateRange

from agent_service.types import PlanRunContext
from agent_service.utils.earnings.earnings_util import (
    get_transcript_partitions_from_db,
    get_transcript_sections_from_partitions,
    insert_transcript_partitions_to_db,
    split_transcript_into_smaller_sections
)
from agent_service.utils.date_utils import (
    get_now_utc
)
from agent_service.tools.earnings import (
    _handle_date_range
)
from dateutil.relativedelta import relativedelta


class TestEarnings(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("asyncio").setLevel(logging.ERROR)

    async def asyncSetUp(self):
        self.context = PlanRunContext.get_dummy()
        self.curr_date = (get_now_utc()).date()

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
        self.assertGreater(len(partition_split.keys()), 1)

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
    
    async def test_handle_date_range(self):
        start_date = datetime(2023, 4, 1).date()
        end_date = datetime(2023, 9, 30).date()

        # Date range is fiscal and quarterly, single stock
        with self.subTest("fiscal_quarterly_single_stock"):
            date_range = DateRange(start_date=start_date, end_date=end_date, is_quarterly=True, is_fiscal=True)
            single_stock = True
            res_start_date, res_end_date, res_fiscal_quarters = await _handle_date_range(
                date_range=date_range, single_stock=single_stock, context=self.context
            )
            self.assertIsNone(res_start_date)
            self.assertIsNone(res_end_date)
            self.assertEqual(res_fiscal_quarters, ['2023Q2', '2023Q3'])
        
        # Date range is not quarterly
        with self.subTest("not_quarterly"):
            date_range = DateRange(start_date=start_date, end_date=end_date, is_quarterly=False, is_fiscal=True)
            single_stock = True
            res_start_date, res_end_date, res_fiscal_quarters = await _handle_date_range(
                date_range=date_range, single_stock=single_stock, context=self.context
            )
            self.assertEqual(res_start_date, start_date)
            self.assertEqual(res_end_date, end_date)
            self.assertIsNone(res_fiscal_quarters)
        
        # 1. Date range is quarterly, not fiscal, excludes_current_date
        # 2. Date range is quarterly, multiple stocks, excludes_current_date
        quarterly_excludes_curr_date_cases = [
            {
                "description": "not_fiscal_quarterly_excludes_curr_date",
                "is_fiscal": False,
                "single_stock": True,
            },
            {
                "description": "fiscal_quarterly_multiple_stock_excludes_curr_date",
                "is_fiscal": True,
                "single_stock": False,
            },
        ]

        # Date range excludes current date
        # Verify start and end dates are pushed forward by 3 months to fetch earnings for the requested quarters 
        # (not earnings reported during those quarters)
        start_date = self.curr_date - relativedelta(months=6)
        end_date = self.curr_date - relativedelta(months=3)

        for case in quarterly_excludes_curr_date_cases:
            with self.subTest(case["description"]):
                date_range = DateRange(
                    start_date=start_date, end_date=end_date, is_quarterly=True, is_fiscal=case["is_fiscal"]
                )
                res_start_date, res_end_date, res_fiscal_quarters = await _handle_date_range(
                    date_range=date_range, single_stock=case["single_stock"], context=self.context
                )
                self.assertEqual(res_start_date, start_date + relativedelta(months=3))
                self.assertEqual(res_end_date, end_date + relativedelta(months=3))
                self.assertIsNone(res_fiscal_quarters)

        # 1. Date range is quarterly, not fiscal, includes_current_date
        # 2. Date range is quarterly, multiple stocks, includes_current_date
        quarterly_includes_curr_date_cases = [
            {
                "description": "not_fiscal_quarterly_includes_curr_date",
                "is_fiscal": False,
                "single_stock": True,
            },
            {
                "description": "fiscal_quarterly_multiple_stock_includes_curr_date",
                "is_fiscal": True,
                "single_stock": False,
            },
        ]

        # Date range includes current date
        # Verify start and end dates are unchanged to fetch earnings reported during the requested quarters
        start_date = self.curr_date - relativedelta(months=3)
        end_date = self.curr_date + relativedelta(months=3)

        for case in quarterly_includes_curr_date_cases:
            with self.subTest(case["description"]):
                date_range = DateRange(
                    start_date=start_date, end_date=end_date, is_quarterly=True, is_fiscal=case["is_fiscal"]
                )
                res_start_date, res_end_date, res_fiscal_quarters = await _handle_date_range(
                    date_range=date_range, single_stock=case["single_stock"], context=self.context
                )
                self.assertEqual(res_start_date, start_date)
                self.assertEqual(res_end_date, end_date)
                self.assertIsNone(res_fiscal_quarters)
        

