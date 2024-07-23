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
    # the numbers used in this function are completely arbitrary
    if isinstance(output, list):
        if len(output) > 0 and len(str(output[0])) > 20:
            if len(output) < 600:
                return pformat(output)

            # pretty printing large lists of objects is rather slow
            begin = output[:250]
            end = output[-250:]
            return "\n".join(
                [
                    f"Output too large: {len(output)}",
                    "First 250:",
                    "[",
                    ",\n".join([pformat(x) for x in begin]),
                    "]",
                    "... End first 250...",
                    f"...skipping: {len(output)-500}...",
                    "...Begin last 250:...",
                    "[",
                    ",\n".join([pformat(x) for x in end]),
                    "]",
                ]
            )

    if isinstance(output, BaseModel):
        return pformat(output)

    # this is a list of numbers or small strings or small objects
    # (or at least the first item was)
    # just print as normal
    return str(output)
