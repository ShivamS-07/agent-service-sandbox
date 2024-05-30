import datetime

from agent_service.io_type_utils import ComplexIOBase, io_type


@io_type
class DateRange(ComplexIOBase):
    start_date: datetime.date
    end_date: datetime.date
