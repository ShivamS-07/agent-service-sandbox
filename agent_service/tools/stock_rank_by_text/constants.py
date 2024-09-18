SAMPLES_TO_SEND_TO_RUBRIC_GEN = (
    3  # Sets the amount of sample company summaries we send to generate the rubric
)
RANDOM_SEED = 421  # Seed for function calls from random library
PAIRWISE_CONCURRENCY = 250
TIEBREAKER_CONCURRENCY = 250
EVALUATE_AND_SUMMARIZE_CONCURRENCY = 150

NONRELEVANT_COMPANY_EXPLANATION = (
    "The documents for this company did not indicate any relevancy towards the subject specified."
)

# Constants to help out with scoring filtered stocks by profile
SCORE_OUTPUT_DELIMITER = "___"
RUBRIC_DELIMITER = "RUBRIC_OUTPUT"
SCORE_MAPPING = {"0": 0.0, "1": 0.2, "2": 0.4, "3": 0.6, "4": 0.8, "5": 1.0}
SAMPLES_DELIMITER = "------"
