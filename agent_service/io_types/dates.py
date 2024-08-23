import datetime

from agent_service.io_type_utils import ComplexIOBase, io_type
from agent_service.io_types.output import Output
from agent_service.io_types.text import Text
from agent_service.utils.boosted_pg import BoostedPG


@io_type
class DateRange(ComplexIOBase):
    start_date: datetime.date
    end_date: datetime.date

    async def to_rich_output(self, pg: BoostedPG, title: str = "") -> Output:
        t: Text = Text(
            val=f"Date range: ({self.start_date.isoformat()}, {self.end_date.isoformat()})"
        )
        return await t.to_rich_output(pg=pg, title=title)
