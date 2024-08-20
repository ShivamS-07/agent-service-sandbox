import re
from typing import Any, List

from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.table import TableColumnMetadata


def update_dataframe(col_label: str, col_values: List[str], df: Any) -> TableColumnMetadata:
    """
    Function Inputs:
    - The label of a column
    - The results of each row in the column, starting from row 1
    - the dataframe which was transformed from the original table

    Function Outputs:
    - Returns a new TableColumn object to represent the column for creating a new table
    - This function will MUTATE the dataframe to remove units off if it returned a TableColumn with units

    This function assumes the following to function:
    - the column values START with an integer/float
    - The column values do not start with . or ,
    - numbers are not directly followed by . or , and instead followed by a space or a letter
    """

    # remove the numbers from the front of each value to get the suffix
    col_values = [str(value) for value in col_values]
    values_set = set(col_values)
    value_suffixes_set = set(
        [re.sub(r"^[\d.,]+", "", value).strip() for value in values_set if value != "n/a"]
    )

    # if the suffix is shared amongst all values, and it is a value which is new (a prefix number was removed)
    if (
        " " not in value_suffixes_set
        and len(value_suffixes_set) == 1
        and len(values_set | value_suffixes_set) > len(values_set)
    ):

        suffix = list(value_suffixes_set)[0]
        value_prefixes = [
            match.group().strip().replace(",", "") if match else "n/a"
            for value in col_values
            for match in [re.match(r"^[\d.,]+", value)]
        ]

        is_float = False
        for value_prefix in value_prefixes:
            is_float = is_float or "." in value_prefix

        if is_float:
            for i, value in enumerate(col_values):
                if re.match(r"^\d", value_prefixes[i]):
                    df.at[i, col_label] = float(value_prefixes[i])

            if suffix == "":
                return TableColumnMetadata(label=col_label, col_type=TableColumnType.FLOAT)
            else:
                return TableColumnMetadata(
                    label=col_label, unit=suffix, col_type=TableColumnType.FLOAT_WITH_UNIT
                )

        for i, data in enumerate(col_values):
            if re.match(r"^\d", value_prefixes[i]):
                df.at[i, col_label] = int(value_prefixes[i])

        if suffix == "":
            return TableColumnMetadata(label=col_label, col_type=TableColumnType.INTEGER)
        else:
            return TableColumnMetadata(
                label=col_label, unit=suffix, col_type=TableColumnType.INTEGER_WITH_UNIT
            )
    else:
        return TableColumnMetadata(label=col_label, col_type=TableColumnType.STRING)
