NAME = "name"
PROFILE_FIRST_CONCURRENCY = 32
PROFILE_SECOND_CONCURRENCY = 8
IMPACT_CONCURRENCY = 10  # no reason for this to be bigger than DEFAULT_IMPACT_RUNS

NUM_IMPACT_INDUSTRY_TABLES = 3  # number of times GPT builds initial table for profiles per polarity
MAX_INDUSTRIES = 3  # max number of industries which will be used in profiles per impact
MIN_INDUSTRIES = 0  # min number of industries used in profiles per impact
MIN_INDUSTRY_RATING = 5.0  # must be at least this rating to be included
GREATER_THAN_AVERAGE = 2.5  # must be at least this much over the average
REPEAT_EFFECT = 0  # repeated industry penalty
IMPORTANCE_POSTFIX = "_imp"
POSITIVE = "positive"
NEGATIVE = "negative"
