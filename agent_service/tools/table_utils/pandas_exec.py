import argparse

# imports for the exec
import datetime  # noqa
import math  # noqa

import numpy as np  # noqa
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Input dataframe as json string.",
    )
    parser.add_argument(
        "-c",
        "--code",
        type=str,
        required=True,
        help="Code file to run to transform dataframe.",
    )
    return parser.parse_args()


def exec_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    local_dict = {"df": df}
    exec(code, globals(), local_dict)
    df = local_dict["df"]
    return df


def write_output(df: pd.DataFrame) -> None:
    print(df.to_json())


def main() -> None:
    args = parse_args()
    df = pd.read_json(args.data)
    # Recreate the index on the first column
    df.set_index(df.columns[0], inplace=True)
    with open(args.code, mode="r") as f:
        code = f.read()
    if "import" in code:
        raise RuntimeError("'import' keyword detected in output code, DO NOT USE 'import'")
    new_df = exec_code(df, code)
    write_output(new_df)


if __name__ == "__main__":
    main()
