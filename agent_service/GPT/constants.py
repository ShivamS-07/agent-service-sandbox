from agent_service.utils.prompt_utils import Prompt

GPT35_TURBO = "gpt-3.5-turbo-0125"
GPT4_TURBO = "gpt-4-turbo-preview"
GPT4 = "gpt-4"
GPT4_O = "gpt-4o-2024-08-06"
GPT4_O_MINI = "gpt-4o-mini"
O1 = "o1-preview"
HAIKU = "claude-3-haiku-20240307"
SONNET = "claude-3-5-sonnet-20241022"
ADA_2 = "text-embedding-ada-002"
TEXT_3_LARGE = "text-embedding-3-large"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_SMART_MODEL = GPT4_TURBO
DEFAULT_CHEAP_MODEL = GPT35_TURBO
DEFAULT_EMBEDDING_MODEL = ADA_2
BEST_EMBEDDING_MODEL = TEXT_3_LARGE

MAX_TOKENS = {
    GPT4_O: 128000,
    GPT4_O_MINI: 128000,
    GPT4_TURBO: 128000,
    GPT35_TURBO: 16384,
    ADA_2: 8192,
}
DEFAULT_OUTPUT_LEN = 4096

OPENAI_ORG_PARAM = "/openai/organization"
OPENAI_API_PARAM = "/openai/api_key"

# 5 minutes, worker does its own retries so we should wait for longer
MAX_GPT_WORKER_TIMEOUT = 20 * 60
TIMEOUTS = {
    GPT4_O: 90,
    GPT4_TURBO: 90,
    GPT35_TURBO: 60,
    GPT4_O_MINI: 60,
    SONNET: 90,
    HAIKU: 60,
    ADA_2: 30,
    TEXT_3_LARGE: 30,
    O1: 60 * 5,
}

JSON_RESPONSE_FORMAT = {"type": "json_object"}
TEXT_RESPONSE_FORMAT = {"type": "text"}

NO_PROMPT = Prompt(name="", template="").format()

FILTER_CONCURRENCY = 100
CHEAP_FILTER_CONCURRENCY = 300


def get_client_name() -> str:
    try:
        with open("/etc/hostname", "r") as f:
            return f.read().strip()
    except Exception:
        return "LOCAL"


def get_client_namespace() -> str:
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            return f.read().strip()
    except Exception:
        return "LOCAL"


CLIENT_NAME = get_client_name()
CLIENT_NAMESPACE = get_client_namespace()
