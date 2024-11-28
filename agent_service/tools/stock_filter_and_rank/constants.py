SAMPLES_TO_SEND_TO_RUBRIC_GEN = (
    3  # Sets the amount of sample company summaries we send to generate the rubric
)
RANDOM_SEED = 421  # Seed for function calls from random library
PAIRWISE_CONCURRENCY = 100
TIEBREAKER_CONCURRENCY = 100
EVALUATE_AND_SUMMARIZE_CONCURRENCY = 150

NONRELEVANT_COMPANY_EXPLANATION = (
    "There is no significant relevant information for this company within the time frame."
)

MIN_STOCKS_FOR_RANKING = 3

# Constants to help out with scoring filtered stocks by profile
SCORE_OUTPUT_DELIMITER = "___"
RUBRIC_DELIMITER = "RUBRIC_OUTPUT"
MAX_RUBRIC_SCORE = 5
SCORE_MAPPING = {"0": 0.0, "1": 0.2, "2": 0.4, "3": 0.6, "4": 0.8, "5": 1.0}
SAMPLES_DELIMITER = "------"

# Update params
MAX_UPDATE_CHECK_RETRIES = 2
UPDATE_REWRITE_RETRIES = 3
