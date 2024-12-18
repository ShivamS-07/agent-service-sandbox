import os
from dataclasses import dataclass

from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    LOCAL_TAG,
    PROD_TAG,
    STAGING_TAG,
    get_environment_tag,
)


# Constants for visible_alpha.py
@dataclass
class UnitsData:
    name: str
    shorthand: str


VA_NO_CURRENCY_PLACEHOLDER = "XXX"

UNITS_LOOKUP = {
    1: UnitsData("Amount", ""),
    2: UnitsData("Percent", "%"),
    3: UnitsData("Days", "Days"),
    4: UnitsData("Years", "Years"),
    6: UnitsData("Number", "#"),
    7: UnitsData("Person-Months", "Person-Months"),
    8: UnitsData("sq ft", "sq ft"),
    9: UnitsData("Ratio", "x"),
    12: UnitsData("Months", "Months"),
    13: UnitsData("Weeks", "Weeks"),
    14: UnitsData("MB", "MB"),
    15: UnitsData("Petabytes", "Petabytes"),
    16: UnitsData("lbs", "lbs"),
    19: UnitsData("Minutes", "Minutes"),
    20: UnitsData("bbl", "bbl"),
    21: UnitsData("bpd", "bpd"),
    22: UnitsData("mmbtu", "mmbtu"),
    24: UnitsData("mcf", "mcf"),
    27: UnitsData("Miles", "Miles"),
    28: UnitsData("hl", "hl"),
    29: UnitsData("MW", "MW"),
    30: UnitsData("mcfe", "mcfe"),
    31: UnitsData("boe", "boe"),
    32: UnitsData("Acres", "Acres"),
    33: UnitsData("Gallons", "Gallons"),
    34: UnitsData("btu", "btu"),
    35: UnitsData("cf", "cf"),
    36: UnitsData("cfe", "cfe"),
    37: UnitsData("GWh", "GWh"),
    38: UnitsData("Metric tons", "Tons"),
    40: UnitsData("km", "km"),
    41: UnitsData("barrel miles", "barrel miles"),
    42: UnitsData("cubic metres", "cubic metres"),
    43: UnitsData("litres", "litres"),
    44: UnitsData("MWh", "MWh"),
    45: UnitsData("Inches", "Inches"),
    46: UnitsData("MVA", "MVA"),
    47: UnitsData("Gbps", "Gbps"),
    48: UnitsData("KWh", "KWh"),
    49: UnitsData("Dekatherms", "Dekatherms"),
    50: UnitsData("cubic yards", "cubic yards"),
    51: UnitsData("hrs", "hrs"),
    52: UnitsData("dwt", "dwt"),
    53: UnitsData("short green tons", "short green tons"),
    54: UnitsData("board feet", "board feet"),
    55: UnitsData("grams", "grams"),
    56: UnitsData("kgs", "kgs"),
    57: UnitsData("troy ounce", "ozt"),
    58: UnitsData("feet", "feet"),
    59: UnitsData("sq mt", "sq mt"),
    60: UnitsData("Metres", "Metres"),
    61: UnitsData("Carat", "Carat"),
    62: UnitsData("Ntk", "Ntk"),
    63: UnitsData("Sq KM", "Sq KM"),
    64: UnitsData("GEO", "GEO"),
    65: UnitsData("kilocalorie", "kcal"),
    66: UnitsData("Joule", "J"),
    67: UnitsData("Bushels", "bu"),
    68: UnitsData("Hundredweight", "cwt"),
    69: UnitsData("Ounce", "oz"),
    70: UnitsData("Gigacalorie", "gcal"),
    71: UnitsData("Short Tonne", "short ton"),
    72: UnitsData("Long Tonne", "long ton"),
    73: UnitsData("Wet Metric Tonne", "wmt"),
    74: UnitsData("Dry Metric Tonne", "dmt"),
    75: UnitsData("Hydraulic horsepower", "hhp"),
    76: UnitsData("Thousand board feet", "mbf"),
    77: UnitsData("Thousand square feet", "msf"),
    78: UnitsData("Thousand linear feet", "mlf"),
    79: UnitsData("Hundred cubic feet", "ccf"),
    80: UnitsData("Gigajoule", "GJ"),
    81: UnitsData("Terajoule", "TJ"),
    82: UnitsData("Petajoule", "PJ"),
    83: UnitsData("Twenty-foot equivalent", "TEU"),
    84: UnitsData("watt", "W"),
    85: UnitsData("Therm", "thm"),
    86: UnitsData("Basis points", "bps"),
    87: UnitsData("TeraHash per second", "TH/s"),
    88: UnitsData("ExaHash per second", "EH/s"),
}
###########################################################################


AGENT_WORKER_QUEUE = os.getenv("AGENT_WORKER_QUEUE", "insights-backend-dev-agent-service-worker")
BOOSTED_DAG_QUEUE = os.getenv("BOOSTED_DAG_QUEUE", "insights-backend-dev-boosted-dag")
AGENT_AUTOMATION_WORKER_QUEUE = os.getenv(
    "AGENT_AUTOMATION_WORKER_QUEUE", "insights-backend-dev-agent-automation-worker"
)
NOTIFICATION_SERVICE_QUEUE = os.getenv("NOTIFICATION_SERVICE_QUEUE", "notification-service-dev")
AGENT_QUALITY_WORKER_QUEUE = os.getenv("AGENT_QUALITY_WORKER_QUEUE", "agent-quality-worker-dev")
# If not set, we will instead use the old boosted-dag queue for sending run_execution_plan
AGENT_RUN_EXECUTION_PLAN_QUEUE = os.getenv("AGENT_RUN_EXECUTION_PLAN_QUEUE", "")


def get_B3_prefix() -> str:
    """
    Returns the appropriate B3 prefix URL based on the current environment.

    Returns:
        str: The B3 prefix URL
    """
    env: str = get_environment_tag()

    if env == PROD_TAG or env == STAGING_TAG:
        return "https://insights.boosted.ai"
    elif env == DEV_TAG or env == LOCAL_TAG:
        return "https://insights-dev.boosted.ai"
    else:
        raise ValueError(f"Unknown environment: {env}")


SUPPORTED_FILE_TYPES = ["Current and Historical Portfolio Holdings"]

DEFAULT_CRON_SCHEDULE = "0 8 * * 1-5"  # Daily at 8am

MEDIA_TO_MIMETYPE = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "plain": "text/plain",
}

CURRENCY_SYMBOL_TO_ISO = {
    "$": "USD",  # US Dollar
    "€": "EUR",  # Euro
    "£": "GBP",  # British Pound Sterling
    "¥": "JPY",  # Japanese Yen
    "₹": "INR",  # Indian Rupee
    "₩": "KRW",  # South Korean Won
    "₽": "RUB",  # Russian Ruble
    "₺": "TRY",  # Turkish Lira
    "R$": "BRL",  # Brazilian Real
    "C$": "CAD",  # Canadian Dollar
}


ISO_CURRENCY_CODES = {
    "AED",
    "AFN",
    "ALL",
    "AMD",
    "ANG",
    "AOA",
    "ARS",
    "AUD",
    "AWG",
    "AZN",
    "BAM",
    "BBD",
    "BDT",
    "BGN",
    "BHD",
    "BIF",
    "BMD",
    "BND",
    "BOB",
    "BOV",
    "BRL",
    "BSD",
    "BTN",
    "BWP",
    "BYN",
    "BZD",
    "CAD",
    "CDF",
    "CHE",
    "CHF",
    "CHW",
    "CLF",
    "CLP",
    "CNY",
    "COP",
    "COU",
    "CRC",
    "CUC",
    "CUP",
    "CVE",
    "CZK",
    "DJF",
    "DKK",
    "DOP",
    "DZD",
    "EGP",
    "ERN",
    "ETB",
    "EUR",
    "FJD",
    "FKP",
    "GBP",
    "GEL",
    "GHS",
    "GIP",
    "GMD",
    "GNF",
    "GTQ",
    "GYD",
    "HKD",
    "HNL",
    "HRK",
    "HTG",
    "HUF",
    "IDR",
    "ILS",
    "INR",
    "IQD",
    "IRR",
    "ISK",
    "JMD",
    "JOD",
    "JPY",
    "KES",
    "KGS",
    "KHR",
    "KMF",
    "KPW",
    "KRW",
    "KWD",
    "KYD",
    "KZT",
    "LAK",
    "LBP",
    "LKR",
    "LRD",
    "LSL",
    "LYD",
    "MAD",
    "MDL",
    "MGA",
    "MKD",
    "MMK",
    "MNT",
    "MOP",
    "MRU",
    "MUR",
    "MVR",
    "MWK",
    "MXN",
    "MXV",
    "MYR",
    "MZN",
    "NAD",
    "NGN",
    "NIO",
    "NOK",
    "NPR",
    "NZD",
    "OMR",
    "PAB",
    "PEN",
    "PGK",
    "PHP",
    "PKR",
    "PLN",
    "PYG",
    "QAR",
    "RON",
    "RSD",
    "RUB",
    "RWF",
    "SAR",
    "SBD",
    "SCR",
    "SDG",
    "SEK",
    "SGD",
    "SHP",
    "SLE",
    "SLL",
    "SOS",
    "SRD",
    "SSP",
    "STN",
    "SVC",
    "SYP",
    "SZL",
    "THB",
    "TJS",
    "TMT",
    "TND",
    "TOP",
    "TRY",
    "TTD",
    "TWD",
    "TZS",
    "UAH",
    "UGX",
    "USD",
    "USN",
    "UYI",
    "UYU",
    "UYW",
    "UZS",
    "VED",
    "VES",
    "VND",
    "VUV",
    "WST",
    "XAF",
    "XAG",
    "XAU",
    "XBA",
    "XBB",
    "XBC",
    "XBD",
    "XCD",
    "XDR",
    "XOF",
    "XPD",
    "XPF",
    "XPT",
    "XSU",
    "XTS",
    "XUA",
    "XXX",
    "YER",
    "ZAR",
    "ZMW",
    "ZWL",
}

MAX_CITABLE_CHANGES_PER_WIDGET = 10
