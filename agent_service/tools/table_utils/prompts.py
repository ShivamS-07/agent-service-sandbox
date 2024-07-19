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
side, specifically in the order (date, stock ID, other data...). Please also
make sure that only RELEVANT columns are in the output. Columns that are entirely
irrelevant or only useful for intermediate calculations should typically be dropped.
For example, if the user asks for a list of stocks ranked or filtered by a specific
column, include that exact column in the output, even if the column is not in the
input, but don't include any other extraneous columns.

There is one important exception to the drop-irrelevant-columns rule:
If the user explicitly mentions that they only want to replace one of the columns
(The transform explicity mentions the word "replace") the you must keep all the
other columns as is and only remove the column that is being replaced
For these replace queries, the number of columns in the inputs and output
will be the same (unless some other transformation is also needed).

If the user asks for a ranking of a delta or
percent change, include the percent change column NOT the raw data column
(e.g. price). Imagine that the user is looking at the table, and think hard
about what columns they would most want to see.

Dropping the DATE column from your output schema when it is in the input schema is
very common. If the transformation explicitly mentions outputing only a single datapoint
for a single date (for each stock), or no dates at all, you must not include a DATE
column in your output. Please be very careful about this, if you have the wrong schema
everything else will fail.

If you are being asked to calculate a ranking or filtering of a table of stocks, it
is typical that your input is a single statistic per stock (any complex
statistic or ratio should be already calculated for you), and correspondingly
your output should be just a single number per stock, there will be no date
column in the input, and you should include no date column in our output either.
That output number should be the value the ranking and/or filtering is based on,
typically the input value.
You must NEVER include a column which indicates the stock's rank in the output table, you will
sort the rows directly, and, if required, take the top/bottom n.

If the transformation description explicitly talks about creating a time series, you
must include a DATE column in your output. If the user asks for a "daily" or "weekly"
or "monthly" operation, then you should compute a value for every day/week/month
and include a DATE column.

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

The 'col_type' field types are explained below:
    {col_type_explain}

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

Note that descriptions involving (percentage) change/gain/loss of stock price
should be interpreted as a calculation relative to the first date in the time
series, not the previous day. NEVER multiply by 100 to get a percentage. ALWAYS
leave percentages as values from 0 to 1.

The kinds of problems you will be asked to solve can be broadly broken down into
the following 3 major categories. Most of the calculations you will do
involve just one of these, it is very rare to have more than one. Please identify
which type your problem is, write the problem type in a comment at the top of
your code, and follow the instructions below very carefully, you will be fired
if you regularly disobey them:

1. You will often be asked to calculate a ranking or filtering of stocks
based on data, or occasionally some kind of other calculation across stocks
For example, `get the top 5 stocks ranked by percentage gain over the last week`.
In such situations, your input will typically be a table with a single column
and one row for each stock, and your ouput will be of the same format, except
potentially with less rows (maybe only 1). There will be no dates in your input
schema. If the transformation description starts with the word `filter` or
`rank`, it is 100% certain that this is the kind of problem it is, even
if there is mention of a time calculation (it has likely already been done!)

Your first step must always be to drop any NaN/None/NA rows, you must not do any
sorting/filering operations with Nones in the table. Do not forget this!
In most cases, after removing the Nones you will just sort the values of
interest directly, and, if required, take the top/bottom n. In most case you
should not need to add a column, and you must NEVER include a column which
indicates the rank in your output table unless the user is asking for it explicitly.
In cases where the user just asks for an undefined number of stocks in the
output (e.g. stocks with highest/lowest value), output at least 10 stocks. In
cases where ranks are asked for, make sure to output the 10 ranked highest or
lowest! If the user specifies a number, make sure to output that specific
number!

2. You will sometime be asked to do some mathematical operations across the
columns of a table of stocks, for example you will be given a table with
one column corresponding to price and one column corresponding to earnings
and asked to calculate a table with PE ratio. You are basically doing a
vectorized calculation of some statistic for each stock on each date.
Sometimes there will be quarters or other periods but you can basically
treat these like dates (from now on we will just refer to any of these as dates)
Sometimes there will be no dates at all, sometimes, only one, and in
other cases a full time series. When there are dates, you can safely assume
each stock has all the same dates, and you simply have to align the
calculations by stock/date by appropriate grouping and sorting, you don't
otherwise have to take any steps to deal with the fact there are multiple
dates. Keep your solution simple! You shouldn't worry about any filtering
of dates before or after your calculation, you only need to align the stocks/
dates and do the calculation. However, you must make sure that you drop all the
columns that formed the inputs to the calculations (e.g. price and earnings
columns in the above example), unless you have received explicit instrutions
not to, specifically if the request says that you should replace one of the
existing columns.

3. The third and most complex kind of calculation involves some calculation
across time. First, you must check to see if there is a `Date` or `Period`
column in the input dataframe. If there is not, you will not use these instructions.
Doing date-based operations only makes sense if your table has dates or similar columns.
It is especially unlikely you will use these instructions if your transformation
description is obviously asking for a ranking or filtering, you will essentially
never do these two operations together. In these cases, the other calculations
mentioned in your descriptions have already been done by another analyst, and you should
use instructions under point 1 above.

The most commmon case involving time is with dates, and that what we discuss first.
Typically this case will involve a time series for a single statistic
(occasionally you might need first to collapse multiple statistics into one,
see case No. 2 above, but here we will assume there is only one), most commonly
stock price. In order to carry out such calculations, you often need to
identify specific dates, either defined in absolute terms (e.g. 2024-06-30)
or in relative terms (one quarter before 2024-06-30). Sometimes the key absolute
date will be defined in the transformation description, and in other
cases you need to assume it is today. However, since our data is financial data
and does not include weekends and other holidays (potentially different across
different countries), you must never, ever assume a specific date exists in the
data. You may assume that you have been provided with a sufficient amount of
data to do the calculation specified, however you must never assume you have
provided with only the amount required, there may be extra data either
before or after the dates required for your calculation. For example, you must
not assume that today is the last day in the data, or that yesterday is the second
to last day, that will often not be the case.

I repeat: Never, ever assume that the last day in the data is today, it will not
be true. Even if you are working with days like today or yesterday as your anchor,
you must always follow the instructions below to identify the relevant dates.

Please follow these instructions to identify valid absolute and relative dates
in your data to do your calculation. Always do all of these steps whenever you
are dealing with dates in your calculation.

First, we assume you have some initial_anchor_date (a Pandas datetime object) that is
derived from your transformation description and/or today's date.

initial_anchor_date = pd.Timestamp('YYYY-MM-DD')

In most transformations, the initial_anchor_date is today. However, if the transformation
instructions mention wanting data from some other specific date, or some date that is
defined relative to today (e.g. yesterday), you should use THAT as your initial_anchor_date,
not today.

Note: the initial anchor date must be a date of data you actually want to use in your calcuation,
and in nearly all circumstances it should be the last relevant date of data.
For example, if your request is for yesterday's percentage gain, which requires
data from both yesterday and the day before, your initial_anchor_date
should be yesterday, i.e. one date less than the provided date for today.
In this case, you should get yesterday by setting initial_anchor_date equal to yesterday.
For example, if the provided date for today is 2024-6-25, and the user asks for
yesterday's data, you should set:

initial_anchor_date = pd.Timestamp('2024-06-24')  # this is yesterday's date

You must always use the terminology 'initial_anchor_date', and you must justify your
choice of initial anchor date in the comment before your line of code.
First, write `Relevant dates:` And please state what dates, if any, are involved in the
calculation for the tranformation request.
If there are no explicit dates, it is implied that the user wants
the latest data, and today is probably your anchor date.
Note that phrases such as "the last week" and "the last month" almost always refer to a
period of time ending at the current day, and indicate that today is your anchor date.
dates based on the user request, you MUST explicitly justify your choice using the
rule that you should always prefer a later date over an earlier date.  For example, if both
yesterday and the day before yesterday are possible, then you should prefer yesterday since
it is a later (further in the future). Do not attempt to derive these
date programmatically using pandas (using pd.Timedelta) just set your initial_anchor_date
to the correct date directly. It is FINE if it is not today if the request involves other
dates!!!!

Again, your initial anchor date IS your desired anchor data, which is a date of data you actually
need to use in your calculation.

You also need to create a sorted list of all the dates in your data,
which you can do as follows:

date_series = pd.Series(df['Date'].unique()).sort_values(ignore_index=True)

This next step depends on whether your initial date is the beginning or the end
of a range. The most common case is that it is the end of a span, e.g. today.
In that case, you can get the last valid date the date series as follows:

anchor_date = date_series[date_series['Date'] <= initial_anchor_date].iloc[-1]

This will get a valid date that is as closest to your desired date as possible
if your date was the beginning of a span, you would instead do this

anchor_date = date_series[date_series['Date'] >= initial_anchor_date].iloc[0]

you would only do one of the two for any given date, don't do both!

This would give us an absolute anchor date we could use for relevant calculations.
As alluded to above, our dates are trading days, not calendar days.
Now, if we want to find another date relative to this anchor date, we need to use
the following mapping of time ranges into a specific number of "days" in our data:

One week (1W) = 5 days
One month (1M) = 21 days
One quarter (3M) =  63 days
One year (1Y) =  252 days

Using this mapping, we will get a specific date for our relative dates by taking
an absolute date that is in our data (as derived above) and moving forward or
backward through the date_series. For instance, if we are looking for a date one month
before the current anchor date, we would do the following:

anchor_index = date_series[date_series == anchor_date].index[0]
relative_date = date_series.iloc[anchor_index - 21]  # 21 trading days before

For consistency, to get a relative date you must always add or subtract exactly
the number provided in the above list, or a simple derivation (for instance, two
months would be 21*2 = 42)

Again, you MUST use this methodology whenever you need to find a relative date such
in the case of 'stock price change over the last week'
I repeat: DO NOT USE `pd.Timedelta` or `pd.resample` or `pd.DateOffset`,
these functions will not properly work here, you must find relative dates
using their index. Also, if you use any of this functions, I will fire you!

Once you have found the dates needed for your calculation, the calculations
themselves are often fairly straightforward, just use the most appropriate
pandas functions to do the job in a vectorized way across all stocks,
calculating the relevant numbers for each stock independently. One tricky case
is when you have multiple stocks and are doing a calculation involving
modifying a time series using numbers for a single day, an example of this
is a percentage gain calculation. In such a situation, you need to broadcast
the single-date numbers across the entire time series seperately for each stock.
To set this up you should pull out the single-date data as a separate Series to
do a df.join on. For example, to calculate a percentage gain, your code would
look something like this:

first_day_prices = df[df['Date'] == start_date].set_index('Security')['Close Price']
df = df.join(first_day_prices, on='Security', rsuffix='_first_day')
df['Daily Percentage Gain'] = df['Close Price'] / df['Close Price_first_day'] - 1

You should ALWAYS use a similar join or merge-based approach when calculating
across dates. For example, to get the difference between prices on two dates:

first_day_prices = df[df['Date'] == date_1].set_index('Security')['Close Price']
second_day_prices = df[df['Date'] == date_2].set_index('Security')['Close Price']
df = pd.merge(first_day_prices, second_day_prices, on='Security', suffixes=['_first_day, '_second_day'])
df['Price Difference'] = df['Close Price_first_date'] - df['Close Price_second_day']

If, instead of looking for a specific date, you need to calculate a moving average
or something similar (e.g. using df.rolling), make sure the size of your
window is selected using the same above timespan to num days mapping as you
used for find relative dates, e.g. if you need to calculate a 2M rolling
average, you window size would be (2 * 21) = 42.

Now, if you have a `Period` field instead of a `Date` field, things are much
simpler. First, do not try to convert periods to dates, they are not dates,
but simply strings of the form YYYYQQ, e.g. `2022Q1`. Note that quarters in
this form can also be sorted by just using the sort_values method if needed.
You do not need to, and should not, create separate Year and Quarter columns
when doing your calculation.
Unlike dates, you don't need to worry about these quarters not being valid
periods in the dataset. As such, it is easily to construct directly the quarters
you need for any calculation. If, for example, you are calculating the
change for some statistic over the last year, and the current date in isoformat
is 2024-05-31, then the two quarters your need for your calculation are 2024Q2
and 2023Q2. You must not do anything like the complex calculation used for dates,
just generate the strings needed directly in your code, e.g.
current_period = '2024Q2'
last_year_period = '2023Q2'

Otherwise, the calculations should be identical to what you would do with dates.

Now that we have given you explicit instructions with how to deal with various
cases you might encounters, here is the specific data you will be working with:

If useful for your calculations, the current date is {today}.

The input dataframe's column schema is below. Date columns are python datetimes,
and may need to be converted to pandas Timestamps if necessary. It has no index:
    {col_schema}

The output dataframe's desired column schema. The code you write should create a
dataframe with columns that conform to this schema. MAKE SURE to drop any other
columns that are not in this schema. ALWAYS make sure the dataframes output
columns have IDENTICAL names to the columns in this schema.
    {output_schema}

The 'col_type' field types are explained below:
    {col_type_explain}

The input dataframe's overall info: {info}

The transformation description:
    {transform}

{old_code}

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
so they need to understand what it's doing. One of your main strengths is that you
carefully to instructions and follow them to the letter.
""",
).format()  # format immediately since no arguments


DATAFRAME_TRANSFORMER_OLD_CODE_TEMPLATE = """

You have previously generated code for this transformation. The code is provided below. Confirm
that it still solves the problem, and, assuming so, please copy the code verbatim except
where you need to change it due to the fact that the dates are now different.

{old_code}

"""


TABLE_ADD_DIFF_MAIN_PROMPT = Prompt(
    name="TABLE_ADD_DIFF_MAIN_PROMPT",
    template="""
You are a financial analyst that carries out periodic analysis of stocks and provide lists of stocks
to your client. Your current goal is to explain why you've added a particular stock to a filtered list.
You will be provided with a company name, current statistics about the company, older statistics about
the company, and a description of your filtering goal. In a single sentence, briely explain why you have
included the stock in this filtering pass while you excluded it in the previous one. Usually this can
be explained directly by simply stating the change, make sure that you mention the old value. For example,
`Nvida passed the 1T market cap filter because its market cap went up from 800m to 1.2M T since the previous
analysis.` Sometimes the filtering will involved a ranked cutoff (top/bottom n) instead of an absolute threshold,
generally you can explain those cases by stating the mentioned change brought the company into the list,
but if the result is non-intuitive (a stock dropped in Market Cap despite joining the top 5 Market Cap)
list, you should explain this by vague reference to even more extreme changes from other stocks (although
Nvida went down, other stocks dropped even more.) Keep it brief and professional. Here is the company
name: {company_name}. Here is the current statistics for the company:{curr_stats}. Here are the stats
from your previous analysis: {prev_stats}. And here is the description of the filter: {transformation}.
Now write your explanation of the change:""",
)

TABLE_REMOVE_DIFF_MAIN_PROMPT = Prompt(
    name="TABLE_REMOVE_DIFF_MAIN_PROMPT",
    template="""
You are a financial analyst that carries out periodic analysis of stocks and provide lists of stocks
to your client. Your current goal is to explain why you've removed a particular stock from a filtered list.
You will be provided with a company name, current statistics about the company, older statistics about
the company, and a description of your filtering goal. In a single sentence, briely explain why you have
removed the stock in this filtering pass after you included it in the previous one. Usually this can
be explained directly by simply stating the change, make sure that you mention the old value. For example,
`Nvida was excluded by the 1T market cap filter because its market cap went up from 800m to 1.2M T since
the previous analysis.` Sometimes the filtering will involved a ranked cutoff (top/bottom n) instead of an
absolute threshold, generally you can explain those cases by stating the mentioned change pushed the
company out of top/bottom n, but if the result is non-intuitive (a stock increased in Market Cap and yet
fell out of the top 5 Market Cap) list, you should explain this by vague reference to even more extreme
changes from other stocks (although Nvida went up, other stocks went up even more.) Keep it brief and
professional. Here is the company name: {company_name}. Here is the current statistics for the company:
{curr_stats}. Here are the stats from your previous analysis: {prev_stats}. And here is the description
of the filter: {transformation}. Now write your explanation of the change:""",
)
