import datetime
from typing import List

from dateutil.relativedelta import relativedelta

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.text import Text
from agent_service.utils.boosted_pg import BoostedPG
from agent_service.utils.date_utils import get_quarters_for_time_range


@io_type
class DateRange(ComplexIOBase):
    start_date: datetime.date
    end_date: datetime.date
    is_quarterly: bool = False
    is_fiscal: bool = False

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        t: Text = Text(
            val=f"Date range: ({self.start_date.isoformat()}, {self.end_date.isoformat()})"
        )
        return await t.to_rich_output(pg=pg, title=title)

    def get_quarters(self) -> List[str]:
        # Note there are hardcoded dependencies on this in earnings tools
        return get_quarters_for_time_range(self.start_date, self.end_date)

    @staticmethod
    def clean_and_convert_str_to_date(date_str: str) -> datetime.date:
        try:
            ret = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError as e:
            if str(e) == "day is out of range for month":
                date_splt = date_str.split("-")
                year = int(date_splt[0])
                month = int(date_splt[1])
                ret = datetime.date(year=year, month=month, day=1) + relativedelta(months=1)
            else:
                raise e
        return ret
