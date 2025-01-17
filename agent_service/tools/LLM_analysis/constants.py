import re

from agent_service.GPT.constants import GPT4_O
from agent_service.io_types.text import StockEarningsSummaryPointText

DEFAULT_LLM = GPT4_O

# These are to try to force the filter to allow some hits but not too many
LLM_FILTER_MAX_PERCENT = 0.2
LLM_FILTER_MIN_TOKENS = 500
LLM_FILTER_MAX_INPUT_PERCENTAGE = 0.75

# Constants to help out with scoring filtered stocks by profile
SCORE_OUTPUT_DELIMITER = "___"
RUBRIC_DELIMITER = "RUBRIC_OUTPUT"
SCORE_MAPPING = {"0": 0.0, "1": 0.2, "2": 0.4, "3": 0.6, "4": 0.8, "5": 1.0}

NO_CITATIONS_DIFF = (
    "Previous evidence which supported the stock's inclusion "
    "is no longer among the documents reviewed in this pass"
)

NO_SUMMARY = "No relevant information"
NO_SUMMARY_FOR_STOCK = "No relevant information for this stock"

ANCHOR_REGEX = re.compile(r" ?\[([a-z]{1,2}),\d{1,2}\]([\.\?\!]?)")
# not worth adding NLTK for just this one case, so just use a regex
SENTENCE_REGEX = re.compile(r"(?<=[^A-Z].[.?!]) +(?=[A-Z])")
ANCHOR_HEADER = "### Anchor Mapping"
CITATION_HEADER = "### Citation Mapping"
CSV_BLOCK = re.compile("```csv([^`]+)```")
JSON_BLOCK = re.compile("```json([^`]+)```")
UNITS = ["thousand", "million", "billion", "trillion"]
CITATION_SNIPPET_BUFFER_LEN = 100
MAX_CITATION_TRIES = 5
MAX_CITATION_CHANGE_RATIO = 0.5
BRAINSTORM_DELIMITER = "!!!"

IDEA = "IDEA"
STOCK_TYPE = "STOCK_TYPE"
TARGET_STOCK = "TARGET_STOCK"

MAX_TEXT_PER_SUMMARIZE = 10000
PREFERRED_TEXT_TYPES = [StockEarningsSummaryPointText]
