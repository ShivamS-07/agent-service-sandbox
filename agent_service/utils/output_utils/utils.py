from devtools import pformat
from pydantic import BaseModel

from agent_service.io_type_utils import Any, ComplexIOBase, IOType
from agent_service.utils.async_utils import gather_with_concurrency


async def io_type_to_gpt_input(io_type: IOType, use_abbreviated_output: bool = True) -> str:
    if isinstance(io_type, ComplexIOBase):
        return await io_type.to_gpt_input(use_abbreviated_output=use_abbreviated_output)
    elif isinstance(io_type, list):
        gpt_inputs = await gather_with_concurrency(
            [io_type_to_gpt_input(val, use_abbreviated_output) for val in io_type]
        )
        return str(list(gpt_inputs))
    return str(io_type)


def output_for_log(output: Any) -> str:
    if isinstance(output, list):
        if len(output) > 0 and len(str(output[0])) > 20:
            return pformat(output)

    if isinstance(output, BaseModel):
        return pformat(output)

    return str(output)
