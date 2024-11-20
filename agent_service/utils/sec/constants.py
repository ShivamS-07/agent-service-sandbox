US = "US"
CUSIP = "cusip"
TICKER = "ticker"
CIK = "cik"
SEC_API_KEY_NAME = "/sec/api_key"

NYSE = "nyse"
NASDAQ = "nasdaq"

FILE_10K = "10-K"
FILE_10Q = "10-Q"
FILE_20F = "20-F"

FILINGS = "filings"
FORM_TYPE = "formType"
COMPANY_NAME = "companyName"
LINK_TO_HTML = "linkToHtml"
LINK_TO_FILING_DETAILS = "linkToFilingDetails"
LINK_TO_TXT = "linkToTxt"
FILED_TIMESTAMP = "filedAt"
DEFAULT_FILINGS_SEARCH_RANGE = 365

DEFAULT_FILING_FORMAT = "text"

MANAGEMENT_SECTION = "managementSection"
MANAGEMENT_SECTION_10K = "7"
MANAGEMENT_SECTION_10Q = "part1item2"

RISK_FACTORS = "riskFactors"
RISK_FACTORS_10K = "1A"
RISK_FACTORS_10Q = "part2item1a"

FILING_DOWNLOAD_LOOKUP = {
    FILE_10K: {MANAGEMENT_SECTION: MANAGEMENT_SECTION_10K, RISK_FACTORS: RISK_FACTORS_10K},
    FILE_10Q: {MANAGEMENT_SECTION: MANAGEMENT_SECTION_10Q, RISK_FACTORS: RISK_FACTORS_10Q},
}

HTML_PARSER = "html.parser"

USD = "USD"
NA_CURRENCIES = ["USD", "CAD"]
EUR_CURRENCIES = ["EUR", "CHF", "GBP"]
