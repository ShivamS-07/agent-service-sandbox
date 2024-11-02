import argparse

# imports for the exec
import datetime  # noqa
import io
import math  # noqa
import sys
import traceback

import numpy as np  # noqa
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_file",
        type=str,
        required=True,
        help="Input dataframe as json file.",
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
    try:
        exec(code, globals(), local_dict)
    except Exception:
        # stack trace from exec doesn't reference the line contents of code
        # only the line number, so lets lookup the line content

        # Get the traceback information
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # get the stack frames
        tb = traceback.extract_tb(exc_traceback)

        line_number = None

        for frame in tb:
            # File "<string>", line 31, in <module>
            if "<string>" in frame.filename:
                line_number = frame.lineno
                break

        if line_number:
            # Get the code that caused the exception
            code_lines = code.splitlines()
            if 0 <= line_number - 1 < len(code_lines):
                error_line = code_lines[line_number - 1]
            else:
                error_line = "<unknown>"

            # Print the exception details and the line content from the exec'd code
            print(f"Exception: {exc_type}: {exc_value}", file=sys.stderr)
            print(f"Line {line_number}: {error_line}", file=sys.stderr)

        raise

    df = local_dict["df"]
    return df


def write_output(df: pd.DataFrame) -> None:
    print(df.to_json(date_format="iso"))


def main() -> None:
    args = parse_args()
    with open(args.code, mode="r") as f:
        code = f.read()
    with open(args.data_file, mode="r") as d:
        serialized_df = d.read()
    serialized_df_io = io.StringIO(serialized_df)
    df = pd.read_json(serialized_df_io)
    if "Year" in df:
        df["Year"] = df["Year"].astype(str)
    if "\nimport " in code or code.startswith("import "):
        raise RuntimeError("'import' keyword detected in output code, DO NOT USE 'import'")
    new_df = exec_code(df, code)
    write_output(new_df)


if __name__ == "__main__":
    main()
