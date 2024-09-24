from agent_service.utils.prompt_utils import Prompt

DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT = Prompt(
    name="DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT",
    template="""
You will be given a json object describing the columns of a pandas
dataframe. You should transform this json into another json which uses the same basic
schema representing a dataframe after the transformation is applied. The transformation
will be described to you. If the output columns are identical to the inputs, you can
simply return an empty list. Note that you should try to keep the table vertically
aligned if possible (that is, more rows than columns).

Use descriptive column names so that someone looking at the list of columns would know
immediately what the table has inside it. Please make sure that the column order
makes sense for a viewer as if it were being viewed in a table. For example, a
date column or a stock ID column (if they are present) should be on the left
side, specifically in the order (date, stock ID, other data...). Please also
make sure that only RELEVANT columns are include in the output. If your transformation
involves the calculation of a new column, input columns whose clear purpose is forming
the basis of that calculation should typically be dropped.

However, if you are not adding a new column but are simply ranking or filtering the data
by a existing column, you must keep all columns in the data as is unless the tranformation
explicitly mentions dropping them. You must NEVER add a column which indicates the stock's
rank in the output table, you will sort the rows directly, and, if required, take the top/bottom n.

Also, if the user explicitly mentions that they only
want to replace one of the columns (The transform explicity mentions the word "replace")
then you must keep all the other columns as is and only remove the column that is being replaced
Thus for ranking, filtering, and replace transformation, the number of columns in the inputs
and output will be the same (unless some other transformation is also needed).

If the user asks for a ranking of a delta or
percent change, include the percent change column NOT the raw data column
(e.g. price). Imagine that the user is looking at the table, and think hard
about what columns they would most want to see.

In nearly all cases you will have a stock column. Typically you will only have
one stock column, but in rare cases (e.g. correlations) you may end up creating two,
do them as two separate columns rather than a single pair.

Dropping the date column from your output columns when it is in the input columns is
very common. If the transformation explicitly mentions outputing only a single datapoint
for a single date (for each stock), or no dates at all, you must not include a DATE
column in your output. Please be very careful about this, if you have the wrong columns
everything else will fail.

If the transformation description explicitly talks about creating a time series, you
must ALWAYS include date column in your output! If the user asks for a "daily" or "weekly"
or "monthly" operation, then you should compute a value for every day/week/month
and include a date column.

If the transformation involves some kind of summing or averaging across stocks, your
output must NOT have a stock output column, you should replace it with a string
output column named 'Stock Group'. Note that the correct col_type for string is
`string` and not `str`!!! The column with the statistic you are aggregating
across stock should be modified in the output to reflect that aggregation, i.e.
`Performance Gain` would become `Average Performance Gain` if you were averaging
across stocks.  Very important: if the transformation description indicates you are
doing an aggregation across stocks (e.g. 'Average performance of stocks ...')
and the input table has a date column, then you MUST preserve that date column in
your output, do NOT drop it. It indicates that the user wants to see a time series,
and if you drop that column then that isn't possible, you will fail on this request
and be fired! Those cases will typically involve outputting three columns: the Date,
the Stock Group, and then the averaged statistic, in that order.

You must NOT change the col_type or units of any particular statistic if you are
simply copying that column from the input to the output. Note that when you divide two
statistics of the same type (e.g. two fields with USD currency), the output is typically
NOT of that type, but rather a percent or a float.

If a date column is required, you must always put it first in your output.

If the transformation description does not relate AT ALL to pandas or any sort
of dataframe transformation, please just keep the columns unchanged. You
should still not output anything other than json.

Below is the json schema of the json you should produce. You should produce a
list of these objects, one for each column. Note that if the column type is
"stock", it means that the column contains stock identifiers. Make SURE to use
the "currency" column type for ANY field that holds a value that represents a
price, currency, or money amount.

JSON Schema:
    {schema}

The 'col_type' field types are explained below, ONLY one of these must be chosen:
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
should still not output anything other than code. You may also do this if you
have no obvious way of mapping the transformation description to the data you
have. Note, however, there are cases where there seems to be missing information
passed to you, but in fact it is possible to proceed under some assumptions,
especially when aggregating across stocks. You must be very conservative about
just rejecting a dataframe entirely! If you do this and you are wrong you will
be fired.

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

In cases where the user clearly wants some kind of filtering (not just ranking!!!!)
but just asks for an undefined number of stocks (e.g. `give me the stocks with the best
P/E`), output at least 10 stocks. If the user specifies a number, make
sure to output that specific number! Don't forget to rank before you apply any filter!

If the user just asks for a ranking without saying anything that suggests a filter,
then you should just rank and return all the stocks. You must not filter unless it is
obvious that the user wants a filter.

Note that when you are ranking or filtering, your input and output columns will almost always
be the same.

If your operation across stocks involves a sum, averaging, or other agglomeration across
stocks, your output will NOT have a stock column (the rows of which correspond to
individual stocks), but rather a string column, often (but not always) with a single row.
At the end of your transformation, you should populate the STRING column with a label
or labels that describe the rows, based on the stock group description included in the
transformation description. Your row label should NOT state the operation you have carried
out, instead it should describe the input stocks (e.g. Healthcare Stocks). Note that when you
do this aggregation, you are responsible ONLY for the aggregation, the stocks will be selected
using a different process and you should accept that it is correct, the purpose of the
description of the relevant stocks in the transformation description is ONLY so you
can provide a proper label to the output row (or rows). You must NOT reject the task simply
because you did not create the list of stocks and do not have the information required
to create the list of stocks, you must assume it already been done for you!

For example, you might be asked to 'get average performance of stocks with P/E greater
than 1' and be provided only a stock and performance column, not a P/E column. In such case
you MUST assume that the list of stocks provided in the current table are exactly those
with P/E greater than 1, and simply average their performance. In this case, the reason
you have been given the information about the stocks is so that you can correctly output the
row label, which in this case would be "Stocks with P/E greater than 1'.

If you are doing an aggregation like averaging across stocks and your output table has
a date column (you are outputting a time series), you must do your averaging across
stocks for each date, and also output a time series. The transformation description
will not necessarily mention this, but you must do it this way unless there
is some explicit indication that you are also aggregating across dates as well.
In this circumstance, your output will have multiple rows, one for each date, and
you must insert the same string label for every row. Again, do not reject this
case simply because it talks about criteria on the stock that you have data for,
and do not just ignore the dates, do the required aggregation across all stocks
for each date. For example, your output table might include an average performance
metric on date 1, an average for date #2, etc. You will accomplish this by grouping
by Date. Again, you will always do aggregation across dates and output a time series
If you have a date column and are not explicitly told otherwise!

Note that you may be sometimes asked to rank by correlation. If you are
doing correlation of stock statistics, you will often end up with a correlation
matrix where the index and columns are both labeled "Security" (or "Stock"), if
this is the case, you MUST change the label for either columns or rows before you
transform the matrix back to a format with two columns (one for each security),
you can't have two columns with the same name!

For example, if you have created a correlation table as such:

df_pivoted = df.pivot(index='Date', columns='Security', values='Close Price')
correlation_matrix = df_pivoted.corr()

You must do to this to the correlation matrix before you proceed:

correlation_matrix.index.name = "Security_1"
correlation_matrix.columns.name = "Security_2"

Do not miss this step, if you directly stack the result it will crash!!!!
You must not set an index in a table where there are two security columns, BTW

If you are doing a more complex calculation such as correlation, don't forget to
rank and/or filter as required by the transformation description!


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

Please follow these instructions to identify valid dates in your data to do your calculation.
Always do all of these steps whenever you are dealing with calculation over dates.

First, before you write code, you must define the specific range of data you need to access in
order to carry out your transformation. State the specific start and end dates of the range
you are calculating over in a comment before you write any code. The start date must always
be before the end date. For a calculation across time, they must NEVER be the same date. Sometimes
the end date is not explicitly defined, but if so it should be today.
For example, if your request is for yesterday's percentage gain, which requires data from both
yesterday and the day before, your initial send date
should be yesterday, and your initial start date would be the day before.
Then, write the exact number of days between the two dates on the following line,
and indicate which of the two methods you must use to derive your method, and what it means

# initial start date is 2023-07-23
# initial end date is 2024-07-24
# days: 366 (Long Ranges method, derive start and end dates separately)

or

# initial start date is 2024-07-23
# initial end date is 2024-07-24
# days: 1 (Short Ranges method, derive start date from end date)

If the number of days between the two dates is greater than 5, you will follow the instructions
immediately below, in the Long Ranges section. If the number is less than or equal to 5, you will follow
other instructions defined in the Short Ranges section

Long Ranges

Assuming the initial start and end dates are at least 5 days apart, create initial start end date objects
for those dates:

initial_start_date = pd.Timestamp('2023-07-23')
initial_end_date = pd.Timestamp('2024-07-24')

You must always use the terminology 'initial_start/end_date', and you must not attempt
to derive these date programmatically using pandas (using pd.Timedelta), you must just set your
initial dates to the correct dates directly.

Next, you will need to create a sorted list of all the dates in your data,
which you can do as follows:

date_series = pd.Series(df['Date'].unique()).sort_values(ignore_index=True)

As we stated above, you can NEVER be certain any particular date you choose is
actually present in the data. In order to get what we call anchor dates (real dates
in the data rather than dates that you have picked), you must always do the follow operations,
you can never use initial dates directly. For the start date, you will find the first date that is
the same or later than your initial start date:

start_anchor_date = date_series[date_series['Date'] >= initial_start_date].iloc[0]

For end date, you will find the last date that is the same or before your initial end date:

end_anchor_date = date_series[date_series['Date'] <= initial_end_date].iloc[-1]

Always use the  >=, iloc[0] version for start dates (start of the range), and always use
the <=, iloc[-1] version for your end dates (end of the range), do not mix them up!

This will get valid dates that are closest to your desired dates as possible.

Short Ranges

If the difference between the start and end date is less than or equal to 5 days,
you will calculate ONLY the end_anchor_date using the Long Range above, i.e.:

initial_end_date = pd.Timestamp('2024-07-24')
date_series = pd.Series(df['Date'].unique()).sort_values(ignore_index=True)
end_anchor_date = date_series[date_series['Date'] <= initial_end_date].iloc[-1]

To get the start_anchor_date, you will get the date by getting the index of the end_anchor_date
and stepping back the number of days between the two dates you noted earlier. If there were
3 days between the two, you will subtract three from the index of the end_anchor_date to get
the start_anchor_date

end_anchor_index = date_series[date_series == end_anchor_date].index[0]
start_anchor_date = date_series.iloc[anchor_index - 3]  # 3 trading days before

DO NOT USE `pd.Timedelta` or `pd.resample` or `pd.DateOffset`, these functions will not properly
work here, in this case, you must find dates using their index. If you use any of this functions,
I will fire you!

Again, you must only do this if the number of days between the start and end dates is no more than
5, but if it is 5 or less, you must find the start_anchor_date in this way, and not the way
described above.

Once you have found reliable anchor dates using the appropriate method, you can access the data
for the specific dates. Again, never, ever use an initial date directly except to convert to an
anchor date using this method.

If your calculation requires you to specificially identify other dates so you can access them
directly in your pandas code, any other anchor dates must be identified using these two methods,
which you should choose based on the proximity to existing dates.

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

first_day_prices = df[df['Date'] == start_anchor_date].set_index('Security')['Close Price']
df = df.join(first_day_prices, on='Security', rsuffix='_first_day')
df['Daily Percentage Gain'] = df['Close Price'] / df['Close Price_first_day'] - 1

You should ALWAYS use a similar join or merge-based approach when calculating
across dates. For example, to get the difference between prices on two dates:

start_day_prices = df[df['Date'] == start_anchor_date].set_index('Security')['Close Price']
end_day_prices = df[df['Date'] == end_anchor_date].set_index('Security')['Close Price']
df = pd.merge(start_day_prices, end_day_prices, on='Security', suffixes=['_first_day, '_second_day'])
df['Price Difference'] = df['Close Price_first_day'] - df['Close Price_second_day']

Other than via merge/join, you must NEVER create a brand new dataframe (with pd.DataFrame)
during this process, it is very easy to mess that up and lose security columns

I repeat, during this process, you must NOT use pd.DataFrame!!!!

As alluded to above, our dates are trading days, not calendar days.

If, instead of looking for a specific date, you need to calculate a moving average
or something similar (e.g. using df.rolling), make sure the size of your
window is selected using the following mapping of calendar periods to trading days:

One week (1W) = 5 days
One month (1M) = 21 days
One quarter (3M) =  63 days
One year (1Y) =  252 days

For example, if you need to calculate a 2M rolling average, you window size
would be (2 * 21) = 42.

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

start_period = '2023Q2'
end_period = '2024Q2'

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
