import datetime

PROPERTY = "property"
EXPLANATION = "explanation"
POLARITY = "polarity"

SUPPORTS = "supported"
CONTRADICTS = "contradicted"
UNRELATED = "unrelated"
SUPPORT_LOOKUP = {SUPPORTS: 1, CONTRADICTS: -1, UNRELATED: 0}

HIGH = "high"
MEDIUM = "medium"
LOW = "low"
SUPPORT_DEGREE_LOOKUP = {LOW: 1.0, MEDIUM: 5.0, HIGH: 10.0}

RELATION = "relation"
STRENGTH = "strength"
RATIONALE = "rationale"
IMPACT = "impact"

NUM_TOPIC_WORKERS = 30

NEWS_TOPICS_BATCH_SIZE = 50
NUM_TOPICS_UB = 100
MAX_BATCHES = 10
TOTAL_RELEVANT_TOPICS_THRESHOLD = 0.1  # each batch should have at least X% relevant topics
IRRELEVANT_TOPICS_THRESHOLD = (
    0.2  # we will stop searching if the bottom X% topics are irrelevant in a batch
)

MIN_MAX_NEWS_COUNT = 5
MIN_MAX_TOP_NEWS_COUNT = 2

ONE_DAY = datetime.timedelta(days=1)
HORIZON_DAY_LOOKUP = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "4M": 120}
HORIZON_DELTA_LOOKUP = {
    horizon: datetime.timedelta(days=days) for horizon, days in HORIZON_DAY_LOOKUP.items()
}


SUPPORT_GPT_WEIGHT = 0.6
SUPPORT_GPT_WEIGHT_EARNINGS = 0.5
SUPPORT_GPT_WEIGHT_EARNINGS_PEERS = 0.5
SUPPORT_NEGATIVE_WEIGHT = 1.25  # give more weight to negative news
SUPPORT_TOP_NEWS_COUNT_WEIGHT = 0.2
SUPPORT_NEWS_COUNT_WEIGHT = 0.2
EARNINGS_PERC_OF_NEWS_COUNT = 0.3  # sets the percentage of max news count for earning topics
SUPPORT_MULTIPLIER = 1.5  # boost scores so they're not too close to zero
NEUTRAL_CUTOFF = 0.2

SUPPORT_STR = (
    "Your argument must indicate that the target situation is true or is more likely be true"
)
OPPOSE_STR = (
    "Your argument must indicate that the target situation is not true or less likely to be true"
)
MIXED_STR = "Your argument should not take a side, but rather present the evidence for both sides objectively."

NUM_NEWS_TOPICS_FOR_SUMMARY = 10
