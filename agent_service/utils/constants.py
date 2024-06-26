import os
from dataclasses import dataclass


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
AGENT_AUTOMATION_WORKER_QUEUE = os.getenv(
    "AGENT_AUTOMATION_WORKER_QUEUE", "insights-backend-dev-agent-automation-worker"
)
