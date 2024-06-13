from agent_service.io_type_utils import ComplexIOBase, IOType


def io_type_to_gpt_input(io_type: IOType, use_abbreviated_output: bool = True) -> str:
    if isinstance(io_type, ComplexIOBase):
        return io_type.to_gpt_input(use_abbreviated_output=use_abbreviated_output)
    elif isinstance(io_type, list):
        return str([io_type_to_gpt_input(val, use_abbreviated_output) for val in io_type])
    return str(io_type)
