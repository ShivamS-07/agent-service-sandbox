from agent_service.GPT.constants import GPT4_O

DEFAULT_LLM = GPT4_O

# These are to try to force the filter to allow some hits but not too many
LLM_FILTER_MAX_PERCENT = 0.2
LLM_FILTER_MIN_PERCENT = 0.05

# Constants to help out with scoring filtered stocks by profile
SCORE_OUTPUT_DELIMITER = "___"
RUBRIC_DELIMITER = "RUBRIC_OUTPUT"
SCORE_MAPPING = {"0": 0.0, "1": 0.2, "2": 0.4, "3": 0.6, "4": 0.8, "5": 1.0}

NO_CITATIONS_DIFF = (
    "Previous evidence which supported the stock's inclusion "
    "is no longer among the documents reviewed in this pass"
)
