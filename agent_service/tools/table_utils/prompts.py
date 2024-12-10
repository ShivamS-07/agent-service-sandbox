from agent_service.io_type_utils import TableColumnType
from agent_service.utils.prompt_utils import Prompt

DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT = Prompt(
    name="DATAFRAME_SCHEMA_GENERATOR_MAIN_PROMPT",
    template="""
You will be given a json object describing the columns of a pandas
dataframe. You should output a json which uses the same basic
schema representing a dataframe after the described transformation is applied.

The simpliest case is when the tranformation involves simply ranking or filtering the data
by a existing column. In this case, you must preserve ALL columns in the input data unless the
tranformation explicitly mentions dropping them. When ranking or filtering, you also must NEVER
add a column which indicates the stock's rank in the output table, you will sort the rows directly,
and, if required, take the top/bottom n.

If the user explicitly mentions that they only want to replace one of the columns
(The transform explicity mentions the word "replace")
then you must keep all the other columns as is and only delete the column that is being replaced.

For most ranking, filtering, and replace transformations, the number of columns in the inputs
and output must be the same (unless some other transformation is also needed). Do not delete columns
not mentioned in the transformation. If you drop data you're not supposed to, you will be fired.

In other cases, you will need to generate a new column. In such a case, use descriptive column names
so that someone looking at the list of columns would know immediately what the table has inside it.
Please make sure that the column order makes sense for a viewer as if it were being viewed in a table.
For example, a date column or a stock ID column (if they are present) should be on the left
side, specifically in the order (date, stock ID, other data...).

If the user asks for a ranking of a delta or
percent change, include the percent change column NOT the raw data column
(e.g. price). Imagine that the user is looking at the table, and think hard
about what columns they would most want to see.

In nearly all cases you will have a stock column. Typically you will only have
one stock column, but in rare cases (e.g. correlations) you may end up creating two,
do them as two separate columns rather than a single pair.
When doing correlation, please pay very careful attention to any mention in the transformation
description of which security column should go first, often the new, derived Security column
will come first. If you put the two security columns in the wrong order, downstream operations
will often fail.

Dropping the date column from your output columns when it is in the input columns is
very common. If the transformation explicitly mentions outputing only a single datapoint
for a single date (for each stock), or no dates at all, you must not include a DATE
column in your output. Please be very careful about this, if you have the wrong columns
everything else will fail.

You should of course drop a column if the transformation explicitly asks you to drop it.
The only other reason to drop a non-date column is when you are creating a new column which will be
directly derived from the data in the existing column. In that case, you should drop the old
column unless told otherwise. For example, if the user is asking for you to calculate return, you
can drop the original stock price column from the input data, and if the user is asking you to
calculate P/E ratio, you can drop both the original price column and the original earnings column.
The only exception to this is when the transformation description explicitly requires you
to keep columns. You must always listen to instructions not to drop columns in the transformation,
this is extremely important.

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

Note you will never, under any circumstances, drop the Security column from the input when it exists,
you will only sometimes replace it with a Stock Group column! Dropping a Security column when you
are not supposed to will result in you being fired.

Another transformation possibility is some calculation which takes daily data (data with a date
column) and converts it to monthly or yearly data. Quarterly data may also be converted to yearly
data. For example, `calculate the stock price growth for each of the last 12 months/5 years`
In such a case, you will replace the date/quarter column with a month/year column in your output
columns. Do this only if the user clearly wants outputs for multiple months/years. If the user
just wants a single datapoint, you should not be including a date-related column at all. Make
sure you change both the type of the column as well as the name/label of the column! Month-type
columns must always be labeled 'Month', and year-type columns must always be labeled `Year`, you
must never use Date or Period as the label for these columns.

If you are changing a Date or Period column to a Month or Year column, you will never, ever touch
the Security column. If there is a Security/stock column in the input, it must be in your output.
Seriously, this is extremely important, do NOT drop the Security column when modifying other columns,
if you do you will be fired! You must listen, damnit!

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

DATAFRAME_TRANSFORMER_SYS_PROMPT_STR = """
You are a quantitative data scientist who is extremely proficient with Python
and Pandas. You will use python and pandas to write code that applies numerical
transformations to a pandas dataframe based on the instructions given to
you. Please comment your code for each step you take, as if the person reading
it was only minimally technical. Your managers sometimes like to see your code,
so they need to understand what it's doing. One of your main strengths is that you
listen carefully to instructions and follow them to the letter.
"""


DATAFRAME_TRANSFORMER_MAIN_PROMPT = Prompt(
    name="DATAFRAME_TRANSFORMER_MAIN_PROMPT",
    template=DATAFRAME_TRANSFORMER_SYS_PROMPT_STR
    + """
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

Note that the values in STOCK columns are not just strings, so you CANNOT
manually add rows with stocks in them.

Your first step must always be to drop any NaN/None/NA rows, you must not do any
sorting/filering operations with Nones in the table. Do not forget this!
In most cases, after removing the Nones you will just sort the values of
interest directly, and, if required, take the top/bottom n. In most case you
should not need to add a column, and you must NEVER include a column which
indicates the rank in your output table unless the user is asking for it explicitly.

In cases where the user indicates they want some kind of filtering (not just ranking!!!!)
but do not provide an explicit formula, instead asking for stocks that are the
best/worst/largest/smallest/etc. for a statistic (e.g. `give me the stocks with the best
P/E`), output at least 10 stocks. If the user specifies a number of outputs that they
want, make sure to output that specific number! Don't forget to rank before you apply
any such filter!

If the user just asks for a ranking without saying anything that suggests a filter,
then you should just rank and return all the stocks. You must not filter unless it is
obvious that the user wants a filter.

Note that when you are ranking or filtering, your input and output columns will almost always
be the same.

There is one kind of filtering you may be asked to do that is more challenging, namely that
you may be asked to filter a stock based on data across multiple dates (including months, quarter
and years). For example: 'filter to stocks which have had at least a 5% stock gain in each of
the last 6 months`. In this case, you must be very careful that you preserve all the data you are
using to do the filtering, in the final table there should only be rows with stocks that pass
the filter, but each stocks will have multiple rows, one for each of the datapoints used in
the filter.

If you have been asked to do filtering involving multiple datapoints per stock (i.e. you are
filtering and there is a date, quarter/period, month, or year column and a clear indication
of more than one datapoint per stock), you must explicit state this in your comment at the
top of your code and follow a strategy that involves deriving a python list corresponding
to the desired stocks, and then filtering the dataframe on that list . After you say it is a
filter problem, say 'This is a multidate filter problem, I will keep data for all relevant
dates for stocks that pass the filter in my output. I will accomplish this by first deriving
a list of stocks to keep, and then filtering the original df on that list.'.

If your operation across stocks involves a sum, averaging, or other agglomeration across
stocks, your output will NOT have a stock column (the rows of which correspond to
individual stocks), but rather a `Stock Group` string column, often (but not always) with
a single row.
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

If the transformation description does not mention what type of stocks are involved or just
mentions 'input stocks', just populate the `Stock Group` column with `input stocks` as the
default.

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

Note that you must never, ever attempt to directly select particular securities.
In the rare case where you need to divide your table up into two or more groups of
stocks (including singleton groups), you will provided with a Group column and
told exactly the string you may filter on to break the table down into the required groups.
The Security columns are not strings and you cannot filter on them to get specific stocks.

Note that if you need to use the apply function to do a calculation over groupings of stocks,
you must remember two particular limitations associated with them:
1. Never, ever use lambdas. You must pass the function directly as an argument of the apply
2. Make sure all the variables you access are defined inside the function

For example, if you had a dataframe df_tech which had the returns for tech stocks and
a dataframe df_spy which had the returns for the SPY, and wanted to calculate the betas
of the tech stocks relative to the S&P 500

df_tech = df_tech.merge(df_spy, on='Date', how='left')

# Group by 'Security' and compute beta
def compute_beta(group):
    covariance = group['Daily Return'].cov(group['SPY Return'])
    variance_spy = group['SPY Return'].var()
    beta = covariance / variance_spy
    return pd.Series({{'Beta': beta}})

beta_df = df_tech.groupby('Security').apply(compute_beta).reset_index()

Note the we use the compute_beta function directly and we first join the spy returns to the df
returns so they are accessible inside the compute_beta function

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

Sometimes you will be converting data broken down by individual dates or quarters
into other units of time, such as months or years. Examples are YoY Revenue growth, or
monthly stock returns. A few important things to keep in mind when doing such calculations:
1. Make sure there is only one output per unit time the user is interested in. To accomplish
this will often involve either filtering to work only with specific dates at the beginning
or end of a period, or summing the values across the time period. If you are not doing one of
those two things, you are probably doing something wrong!
2. When deciding to filter or sum, you must consider the nature of the data. Stock price
and many statistics related to it cannot be summed, it does not make sense. Overall debt
is similar. However, raw revenue and other income-related values already refects a delta
for a given period, so summing makes sense (you get the total revenue for a year by summing
the revenue for the quarters, but you do NOT get a 'total' stock price for a year by summing
stock price on each day).
3. Note that if you are doing a monthly/yearly change calculation with daily data, you must explicitly
filter down to just the last day of each month/year, you should sort the dates, groupby the security
AND month/year columns and then use tail to take the last day for each month.
4. Note that you can only do proper year-over-year (or month-over-month) growth calculations
when you have complete years (months). You must exclude partial years (months) from your
calculation. For example, if the current date is in 2024, you cannot include year-over-year
revenue growth for 2024 in your output.
5. Make sure you output the exact number of outputs the user wants, no more, no less. For example,
if the user is asking for 5 years, you must output 5 datapoints. Remember you usually need an extra
datapoint because you must throw away the partial month/year we are currently in, for instance if
you are in 2024 you will need your output years to be 2019, 2020, 2021, 2022, and 2023.
However, note that if you are calculating YoY you will need an extra year of data to do the calculation,
in this example 2018. For monthly calculations, you will list the relevant months (e.g. 2024-01, 2024-02...)
You should assume you have all the input data you need to do the calculation, but you might have
extra data, so make sure you are explicitly filtering down to exactly the datapoints you want to output
at the end of the process (Do not do this at the beginning and remove data you need for your calculation)
If you are doing a monthly/yearly calculation, in addition to following the procedure for a
'calculation across time' problem as discussed above, you must have another comment at the top of your code
that states that fact ('this is also a monthy/yearly conversion problem), and then explicitly enumerates
your output months/years. If it is a YoY or other such problem that requires additional data for the
calcuation,please state that you need to use that data in the comment before you start coding. You must
always have a comment like this when you write this kind of code or otherwise you are likely to make a mistake.
Please also mention in the comment for the code step that filters down to exactly the required output
datapoints that you are about to do this filtering, do not forget!



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

{error}
Write your code below. Make sure you reassign the output to the same variable `df`.
Your code:

""",
)


DATAFRAME_TRANSFORMER_SYS_PROMPT = Prompt(
    name="DATAFRAME_TRANSFORMER_SYS_PROMPT",
    template=DATAFRAME_TRANSFORMER_SYS_PROMPT_STR,
).format()  # format immediately since no arguments


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

PICK_GPT_MAIN_PROMPT = Prompt(
    name="PICK_GPT_MAIN_PROMPT",
    template="""
You are a senior financial analyst that is deciding whether to give a small pandas coding task to
an intern or an salaried employee. You don't trust the intern's skills yet, in particular you would
only want to give them simple tasks that involve simple filtering or ranking filtering stocks based
on a single calculated measure for each stock, or simple stock return (cumulative stock price change)
calculations. More challenging tasks, including any task that could possibly involve using the groupby
function or otherwise manipulating the columns of the table, should be left to experts.

You should default to filtering or ranking tasks being easy. It does not usually matter what we are filtering
or ranking by, because this will already be calculated at this point. The one key exception is to this
rule is filtering tasks that clearly involve multiple datapoints per stock, for example `filter to
stocks which have 5% return in each of the last 6 months`. In those cases, you will need to groupby
stocks, and the task is therefore hard. However, simple cases like `filter to stocks that have 5% weekly
return` should be considered easy.

Similarly, you should default to easy for most basic calculations of stock performance (simple cumulative
or daily returns/stock price changes), e.g. `calculate percentage price gain for TSX stocks for the 6 months`,
however if the task involves calculation of monthly, quarterly, or yearly returns involving multiple months,
quarters, or years (e.g. `calculate performance for TSX stocks for each of the last 6 months`), you should
consider the task hard.

Simple calculations of minimum or maximum value for a statistic over a time range should also
be considered easy.

All other calculations that are not otherwise trivial should be considered hard.

Here is the task: {task}

Output the word 'easy' if you think that the task is simple enough for the pandas newbie intern, or
'hard' if you think it is not trivial. Output only one of those two words:

""",
)

UPDATE_DATAFRAME_TRANSFORMER_MAIN_PROMPT_STR = "You are financial analyst in charge of a script which carries out a important calculation by manipulating a pandas dataframe. Your current task is to update an older version of the script by changing any date-relevant aspects of the script to use the newer dates, as required. You will be provided with the old code, the old description of the transformation, an updated description (which might be unchanged), the date the last script was run (which might be None; if it is, you should try to infer the previous date from the other information), and the current date. Unless there is nothing to change, you should copy verbatim everything from the original script (which ran successfully) but switch out any references to dates in either the code or the comments that need to be updated based on the differences between the old description and the current description, or the old date and the current date, or both. I repeat, do not make any non-date related changes to the script, and you will generally only switch out string literals or references to dates in comments. Note that in addition to dates in YYYY-MM-DD format, you may also need to change months (YYYY-MM), quarters (YYYYQQ, e.g. 2024Q4), or years, all of which must ultimately be expressed as string literals. Depending on the transformation, you may or may not need to use the current date, and you may not need to do anything at all; be careful not to make unnecessary changes, if you do you will be severely punished. If there is nothing to change because the code does not mention any kind of date, you can simply output the words 'No Change' and nothing else. Otherwise you should rewrite the entire code from beginning to end, do not leave anything out. Here is the old_description: `{old_description}` Here is the new description: `{description}`. Here is the old date (if any): {old_date}. Here is the new date: {date}. Here is the old code, delimited by ---\n:\n---\n{old_code}\n---\nNow write the updated code, or `no change` if no change is needed:\n"  # noqa: E501

UPDATE_DATAFRAME_TRANSFORMER_MAIN_PROMPT = Prompt(
    name="UPDATE_DATAFRAME_TRANSFORMER_MAIN_PROMPT",
    template=UPDATE_DATAFRAME_TRANSFORMER_MAIN_PROMPT_STR,
)

TEXT_TO_TABLE_CITATION_PROMPT_STR = """

Information in your output text that is derived from sources must be cited, you may use your
judgement to determine if a specific claim must be cited, but err on the side of more
citations. Individual texts in your collection of sources are delimited by ***, and each one starts
with a Text Number and mentions the type. Follow the instructions below to do proper citation, which
consists of two major steps. The first step is OPTIONAL, and should ONLY be done in cases where
you're pulling together a lot of disparate data into a single table. If you are directly citing a
table that is in a text, and just "copying" it, you may skip step one.

Step 1 (do not do this if you are citing an existing table directly):
When you are writing your text, at the end of each CSV value (in every row and every column) which
contains information taken from one of the provided source texts, you may output one and only one
citation anchor. Note that a CSV cell is a SINGLE value within a row. A citation anchor consists of
a space before the preceding cell value, an opening square bracket, an anchor id of exactly two
letters, a comma, a one digit whole number indicating the count of citations to be inserted at that
point, a close square bracket, and then a period or newline. To demonstrate, here is an example of
some csv text with citation anchors:

Date Col (date),Header Col 1 (string),Header Col 2 (integer)
"2024-01-01","some value [aa,1]", "1234 [ab,3]"

Notice how the citation anchors come inside the quotes, but always at the end of each cell. The 1
indicates a single citation needs to be inserted at this point [aa,1] and the 3 indicates there are
3 corresponding citations for [ab,3]. The next one would have anchor id `ad` [ad,2]. Each citation
anchor you insert will use a different 2 letter identifier (e.g. [aa,1], [ab,1]...[ba,1],
[bb,1]...[zy,1], [zz,1]). The citation ids in citation anchors are unique identifiers for each
anchor citation determined by the ordering of anchors in the output, they do not have any other
meaning, in particular they are NOT associated with the source texts in any way. The first anchor
appearing in your output csv will always have the anchor id `aa`, the second anchor citation will
have an anchor id of `ab`, the third `ac`, etc. Again, it is a unique identifier, so you may use the
letter(s) in each citation anchor (or letter pair) only once! That is, if you have one sentence with
[aa,1] at the end, you MUST NOT have another anchor citation with just `aa` as the citation letter;
even if you cite the same source text again later, you will use a different letter for the anchor
id! If you reuse the letters in anchor citations across multiple anchor citations we will fail to
parse the document and you will be fired. Again, the number in the citation anchor (the 3 in [ab,3]
indicates the COUNT of distinct, non-overlapping text snippets from the source documents that you
are citing at this anchor. It is NOT the associated Text Number(s) for the sources, which you will
indicate in Step 2. If you output the Text Number of any source text during the first step (when you
are creating the csv), we will fail to parse the document and you will be fired. The citation count
must be greater than 0, and no more than 5. Very Important: the citation anchor is a placeholder for
one or more citations, it is not a citation itself, and so you must insert only one citation anchor
at the end of any csv cell, you must use a whole number to indicate the count of citations at that
point, and then indicate the specific citations at the end of the document, in the anchor
mapping. Most cells in your output csv will likely have a citation anchor, but do not insert a
citation anchor for those that do not provide some specific information taken from one of the
numbered source texts. At this stage, you are not providing any information about which specific
source you have used. Remember that you can cite ANY csv cells (not just rows)!! Use your best
judgement for which cells require citations, but err on the side of more citations.
If possible, you should try to skip step 1 and cite an existing table directly.

Step 2 (always required):
 After you have completed writing your csv output, on the final line of your output, you will output
a json mapping (called the anchor mapping) which maps from the two letter anchor ids used in your
citation anchors to a list of json mappings, where each of these mappings (called citation mappings)
corresponds to an individual citation. If you skipped step 1, simply make up anchor ID's as they
don't matter. Since every anchor corresponds to a situation where at least one source text was used,
you should have at least one citation per citation anchor, and you will often have more. Your list
of citations for each anchor (again, there is only one anchor per cell!) must directly relate to the
value being shown, however if two citations cover the same simple facts, you must only include
one. You should generally have the same number of citations in your list as you indicated in the
corresponding anchor in your csv output. Generally, however, more citations under a single anchor is
NOT better unless they are providing important new information, and you must never, ever have more
than 5 for each citation anchor. Every citation mapping will have a `num` key, the value of which is
an integer which is the Text Number of the source text for the citation. When the source text is
news, e.g. News Development Summaries or News Articles, which are typically only a couple of
sentences long, you will include only the `num` key. For instance, if you have just one anchor whose
preceding sentence contains information from two news topics (i.e [aa,2]) which are tagged as Text
Numbers 3 and 6, your anchor mapping would look like this: {{"aa":[{{"num":3}}, {{"num":6}}]}}. You
will also do this when citing a single table, if you have skipped step 1.

Note that values of the anchor mapping must be lists even if you have only one citation for your
anchor. If you are citing this way, you may only cite a source text once at the anchor point, do not
include multiple citations of the same text at the same anchor point if the citation mappings are
the same. However, for longer source texts you will need to specifically indicate the part of the
source text that provides the information that you are using (the ONLY exception to this is why you
are citing the text to say that it did NOT contain information the client was looking for; if so,
leave the snippet key out). To do this, you will include another key, `snippet`, whose value will be
a single short snippet copied verbatim (word for word) from a single contiguous span of the relevant
source text. This should be the span of text from the source text that best expresses the
information that you used in your csv output from this source text in the cell which ends at the
anchor point, however it is even more important it is an exact copy of a span of text in the source
text. When you are outputting a snippet for your citation, you must output the snippet first, and
then the num corresponding to the Text Number; after you output the snippet, please double check the
Text Number for the source document the snippet came from and make 100% sure you have selected the
correct Text Number for that snippet. If it is still missing key information used in the associated
sentence in your output, you should add another citation mapping from the same text. Again, it is
okay to add more citations that you originally planned if you need to fully cover the information in
your output csv, as long as the citations are not redundant. If the csv cell you produced contains
an amount (i.e. a dollar value), your snippet must include that amount. If the number is only
available in a table, your snippet may consist of simply that number from the table, nothing else,
but if the exact number you need to cite from a table is also in the text, you must cite the text
and not the table. You should avoid including more than two sentences in your snippet. Again, you
must copy your snippet exactly verbatim from a single contiguous span of the source text. Each time
you add a word to your snippet, you must look back at the part of the source text you are copying
from and be 100% sure you are adding exactly the same next word, you may only add what appears next
in the original source text and you must NEVER leave anything out once you start copying. The ONLY
change you may make is when you need to escape characters to make a valid json string (for example,
you\'ll need to add a \\ before any "). Important: after the snippet string is loaded, it must be an
exact substring of the original source text. If it isn\'t, the entire citation project will fail and
you will be fired. Your snippet must be entirely identical to the original source text, down to the
individual punctuation marks; you must only copy and never combine, shorten, reword or reorganize
the snippet, you must not even add or modify a single punctuation mark, or change
capitalization. Even if there is a formatting error in the source text, you must copy that error. It
is absolutely critical that you preserve the original span of text from the source exactly, down to
each individual character of your snippet. It must be a perfect copy. Also, you must never, ever
include a snippet that includes a newline, your anchor json mapping must always be on a single line
(Do not delete newlines, you must just avoid selecting snippets which include them, stop copying
when you hit one).  If you are including a snippet, you may include multiple citations of the same
source text at a single anchor as long as they provide distinct information and there is no overlap
in their snippet strings (but they must be separate citations, do not add their snippets together
into a single citation!). You must always include a snippet when you pull information from any
source text larger than a paragraph, and you must never include a snippet for smaller texts
(news). You should never, ever include a snippet that is simply the entire source text. Don\'t
forget to provide at least one citation mapping for each of the anchors, the anchor mapping should
have exactly as many key/value parts as there are anchors in your csv! You must never, ever leave
out the anchor mapping, if you have no anchors (and hence no citations), output a totally mapping
citation mapping. Again your anchor mapping (which contains any citation mappings) must be a valid
json object on a single line. You must never, ever add any kind of header before the anchor mapping
even if there are other headers in the document (it is not part of your text and will be removed
from it when we parse) and you must not add any other wrappers (no ```json!!!)."""

TEXT_TO_TABLE_MAIN_PROMPT = Prompt(
    name="TEXT_TO_TABLE_MAIN_PROMPT",
    template="""
You are a financial analyst tasked with synthesizing a number of texts into a csv table. The csv
output should have a header line with column titles. ALL csv cells should be wrapped in quotes,
ESPECIALLY if they have commas inside them already. Citation anchors should appear INSIDE of quotes
for each cell!!!! Your header cells should also always contain their type in parentheses! If there's
no data for what is asked for, you should NEVER output an empty cell, always write something even if
it's just telling the user there's no data. You will be fired if you do not follow these instructions.

The description of the table you should create is:
    {table_description}

Here are the document(s), delimited by -----:
-----
{texts}
-----
Here is the transcript of your interaction with the client, delimited by ----:
----
{chat_context}
----

For reference, today's date is {today}.
""",
)

TEXT_TO_TABLE_SYS_PROMPT = Prompt(
    name="TEXT_TO_TABLE_SYS_PROMPT",
    template=f"""
You are a financial analyst tasked with synthesizing a number of texts into a csv table. The csv
output should have a header line with column titles.

After each column title, put the column type in parentheses. The list of all possible columm types
are below. Try to make the csv data match as closely as possible to the expected type (for example,
a price delta or something similar should be normalized to between zero and one before using the
'delta' column type). Never ever choose two column types, just do your best and choose the one the
fits the best. Column types:
{TableColumnType.get_type_explanations()}

{TEXT_TO_TABLE_CITATION_PROMPT_STR}
""",
)
