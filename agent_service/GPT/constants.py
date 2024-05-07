GPT35_TURBO = "gpt-3.5-turbo-0125"
GPT4_TURBO = "gpt-4-turbo-preview"
GPT4 = "gpt-4"
SONNET = "claude-3-sonnet-20240229"
ADA_2 = "text-embedding-ada-002"
TEXT_3_LARGE = "text-embedding-3-large"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_SMART_MODEL = GPT4_TURBO
DEFAULT_CHEAP_MODEL = GPT35_TURBO
DEFAULT_EMBEDDING_MODEL = ADA_2
BEST_EMBEDDING_MODEL = TEXT_3_LARGE

MAX_TOKENS = {GPT4_TURBO: 128000, GPT35_TURBO: 16384, ADA_2: 8192}
DEFAULT_OUTPUT_LEN = 1024

OPENAI_ORG_PARAM = "/openai/organization"
OPENAI_API_PARAM = "/openai/api_key"

# 20 minutes, worker does its own retries so we should wait for longer
MAX_GPT_WORKER_TIMEOUT = 20 * 60
TIMEOUTS = {GPT4_TURBO: 90, GPT35_TURBO: 60}

JSON_RESPONSE_FORMAT = {"type": "json_object"}
TEXT_RESPONSE_FORMAT = {"type": "text"}
