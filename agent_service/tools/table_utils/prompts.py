from agent_service.utils.prompt_utils import Prompt

DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT = Prompt(
    name="DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT",
    template="""
You will be given a json object describing the columns of a pandas
dataframe. You should transform this json into another json with the same schema
representing a dataframe after a transformation is applied. The transformation
will be described to you, and you should produce only the json output for the
new columns. If the output columns are identical to the inputs, simply return an
empty list. Note that you should try to keep the table vertically aligned if
possible (that is, more rows than columns).

Use descriptive column names so that someone looking at the schema would know
immediately what the table has inside it. Please make sure that the column order
makes sense for a viewer as if it were being viewed in a table. For example, a
date column or a stock ID column (if they are present) should be on the left
side, specifically in the order (date, stock ID, other data...).

If the transformation description does not relate AT ALL to pandas or any sort
of dataframe transformation, please just return the dataframe unchanged. You
should still not output anything other than json.

Below is the json schema of the json you should produce. You should produce a
list of these objects, one for each column. Note that if the column type is
"stock", it means that the column contains stock identifiers. Make SURE to use
the "currency" column type for ANY field that holds a value that represents a
price, currency, or money amount.

JSON Schema:

    {schema}

The transformation that will be applied to the dataframe is:
    {transform}

Here is the json describing the input dataframe, one object per column:
    {input_cols}

Please produce your json in the same format describing the columns after the
transformation has been applied. Please produce ONLY this json.

{error}
""",
)

DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT = Prompt(
    name="DATAFRAME_SCHEMA_GENERATOR_SYS_PROMPT",
    template="""
You are a quantitative data scientist who is extremely proficient at both finance and coding.
""",
).format()


DATAFRAME_TRANSFORMER_MAIN_PROMPT = Prompt(
    name="DATAFRAME_TRANSFORMER_MAIN_PROMPT",
    template="""
You will be given the description of a pandas dataframe stored in the variable
`df`, as well as a description of a transformation to apply to the
dataframe. You will produce ONLY python code that utilizes pandas to apply these
transformations to the dataframe. The resultant dataframe must also be called
`df`. The variable name must be kept the same, however the data stored in the
variable `df` should be what the user asks for. The data inside `df` can be
transformed arbitrarily to match the desired output.

You can assume the following code is already written:

    import datetime
    import math

    import numpy as np
    import pandas as pd

    df = pd.DataFrame(...)  # input dataframe

Write only the continuation for this code with no preamble or follow up text. DO
NOT INCLUDE ANY OTHER IMPORTS.

Make 100% sure that you refer to the input variable as `df`, it is defined for
you as above.

If the transformation description does not relate AT ALL to pandas or any sort
of dataframe transformation, please just return the dataframe unchanged. You
should still not output anything other than code.

The input dataframe's column schema is below. Date columns are python datetimes,
and may need to be converted to pandas Timestamps if necessary. It has no index:
    {col_schema}

The output dataframe's desired column schema. The code you write should create a
dataframe with columns that conform to this schema.
    {output_schema}

The input dataframe's overall info: {info}

The transformation description:
    {transform}

{error}
Write your code below. Make sure you reassign the output to the same variable `df`.
Your code:

""",
)

DATAFRAME_TRANSFORMER_SYS_PROMPT = Prompt(
    name="DATAFRAME_TRANSFORMER_SYS_PROMPT",
    template="""
You are a quantitative data scientist who is extremely proficient with Python
and Pandas. You will use python and pandas to write code that applies numerical
transformations to a pandas dataframe based on the instructions given to
you. Please comment your code for each step you take, as if the person reading
it was only minimally technical. Your managers sometimes like to see your code,
so they need to understand what it's doing.
""",
).format()  # format immediately since no arguments
