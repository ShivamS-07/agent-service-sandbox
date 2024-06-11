import datetime
import html
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import backoff
from gbi_common_py_utils.utils.ssm import get_param
from sec_api import ExtractorApi, MappingApi, QueryApi, RenderApi  # type: ignore

from agent_service.utils.async_utils import async_wrap
from agent_service.utils.sec.constants import (
    CIK,
    COMPANY_NAME,
    CUSIP,
    DEFAULT_FILING_FORMAT,
    FILE_10K,
    FILE_10Q,
    FILING_DOWNLOAD_LOOKUP,
    FILINGS,
    FORM_TYPE,
    LINK_TO_HTML,
    NASDAQ,
    NYSE,
    SEC_API_KEY_NAME,
    TICKER,
    US,
)
from agent_service.utils.sec.supported_types import SUPPORTED_TYPE_MAPPING

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=True)
class SecurityMetadata:
    gbi_id: int
    isin: str
    ticker: str
    currency: str
    company_name: str
    security_region: str


class SecMapping:
    """
    The code is copied from NLPService repo
    """

    mapping_api = MappingApi(get_param(SEC_API_KEY_NAME))

    @classmethod
    @backoff.on_exception(backoff.expo, Exception, max_time=60)
    def _resolve_sec_mapping(cls, param: str, value: str) -> Any:
        return cls.mapping_api.resolve(param, value)

    @staticmethod
    def _can_map_isin_to_cik(isin: str) -> bool:
        """
        If the ISIN is a 'US' ISIN (ie starts with 'US') then it can be
        converted to CUSIP and then CIK. Otherwise, we need to try a more
        roundabout method for getting the CIK via the ticker and company name.
        """
        return isin.startswith(US)

    @classmethod
    @async_wrap
    def map_gbi_id_to_cik(
        cls, gbi_id: int, gbi_id_metadata_map: Dict[int, SecurityMetadata]
    ) -> Optional[str]:
        """Map GBI ID to SEC CIK via SEC Mapping API
        NOTE that because the SEC API is kind of slow, we would do 1 gbi_id at a time with the
        `gbi_to_isin_lookup` loaded already (through DB call)

        Args:
            gbi_id (int): target GBI ID
            gbi_to_isin_lookup (Dict[int, str]): GBI ID to ISIN lookup

        Returns:
            Optional[str]: stock's SEC CIK or None if not found
        """
        metadata = gbi_id_metadata_map.get(gbi_id)
        if not metadata:
            logger.error(f"Invalid gbi_id: {gbi_id} not in our system...")
            return None

        if cls._can_map_isin_to_cik(metadata.isin):
            return cls._map_isin_to_cik(metadata.isin)

        # If the ISIN is non-US or not found, try a more roundabout method...
        return cls._map_non_us_isin_to_cik(metadata)

    @classmethod
    def _map_non_us_isin_to_cik(cls, sec_meta: SecurityMetadata) -> Optional[str]:
        # full example of response values: https://sec-api.io/docs/mapping-api/python-example
        # we want to get exact matches only, thus the regex characters below
        results = cls._resolve_sec_mapping(TICKER, f"^{sec_meta.ticker}$")

        for item in results:
            if not item:
                continue

            # skip if not using the two primary USA stock exchanges
            exchange = item["exchange"].lower()
            if NYSE not in exchange and NASDAQ not in exchange:
                continue

            # skip if the wrong currency
            if item["currency"] != sec_meta.currency:
                continue

            # skip if delisted
            if item["isDelisted"]:
                continue

            # if we've passed all the above checks, this CIK works
            return item[CIK]

        # if we went through all results without finding anything, return None
        return None

    @classmethod
    def _map_isin_to_cik(cls, isin: str) -> Optional[str]:
        cusip = cls._convert_isin_to_cusip(isin)
        if cusip is None:
            return None

        return cls._convert_cusip_to_cik(cusip)

    @staticmethod
    def _convert_isin_to_cusip(isin: str) -> Optional[str]:
        """ISIN codes include 12 characters starting with country code and ending with a check digit.
        We only need the 9 characters in the middle.

        Args:
            isin (str): 12-character string

        Returns:
            Optional[str]: the middle 9 characters if the ISIN is valid, else None
        """
        if not isin or len(isin) != 12 or not isin.startswith(US):
            return None
        return isin[2:-1]

    @classmethod
    def _convert_cusip_to_cik(cls, cusip: Optional[str]) -> Optional[str]:
        """The API itself has a retry logic (3 times) when the status code is 429
        Otherwise it will raise Exception

        NOTE: It's kind of slow locally. 100 gbi_ids took ~15s to run.
        TODO: Store the mapping into DB and directly fetch from it

        Args:
            cusip (Optional[str]): cusip code

        Returns:
            Optional[str]: cik code
        """
        if not cusip:
            return None

        try:
            # A list of dictionary with the stock's metadata
            result: List[Dict] = cls._resolve_sec_mapping(CUSIP, cusip)
        except Exception as e:
            # "API error: {} - {}".format(response.status_code, response.text)
            logger.warning(e)
            return None
        else:
            return result[0][CIK] if len(result) == 1 else None


class SecFiling:
    """The class use QueryApi to find out the URLs to the latest 10K/10Q files for a given CIK
    The use ExtractorApi to download the management section of the latest 10K/10Q file
    """

    api_key = get_param(SEC_API_KEY_NAME)
    query_api = QueryApi(api_key)  # use it to get metadata (e.g. URL) of the filings
    extractor_api = ExtractorApi(api_key)  # use it to extract the sections from the filings
    render_api = RenderApi(api_key)  # use it to download the full content of filings

    file_type_10kq = {FILE_10K, FILE_10Q}

    MAX_SEC_QUERY_SIZE = 50

    @classmethod
    def build_query_for_10k_10q_filings(
        cls,
        cik: str,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> Dict:
        """Build the query string for the SEC Filing API

        Args:
            cik (str): Stock's CIK code

        Returns:
            Dict: A query in dictionary format. To download the filing via the query, call
        `SecFiling.query_api.get_filings(query)`
        """
        if end_date is None:
            end_date = datetime.datetime.today() + datetime.timedelta(days=1)
        if start_date is None:
            # default to a quarter of data (one filing)
            start_date = end_date - datetime.timedelta(days=90)

        end_date_str = end_date.isoformat()
        start_date_str = start_date.isoformat()

        forms = " OR ".join((f'formType:"{typ}"' for typ in cls.file_type_10kq))

        query = {
            "query": {
                "query_string": {
                    "query": f"cik:{cik} AND filedAt:[{start_date_str} TO {end_date_str}] AND ({forms})"  # noqa
                }
            },
            "from": "0",
            "size": str(cls.MAX_SEC_QUERY_SIZE),
            "sort": [{"filedAt": {"order": "desc"}}],
        }
        return query

    @classmethod
    def download_10k_10q_section(cls, filing: Dict, section: str) -> Optional[str]:
        """Download 10K/10Q section from sec-api.io

        Args:
            filing (Dict): A dictionary with the following fields: ['id', 'accessionNo', 'cik',
        'ticker', 'companyName', 'companyNameLong', 'formType', 'description', 'filedAt',
        'linkToTxt', 'linkToHtml', 'linkToXbrl', 'linkToFilingDetails', 'entities', 'periodOfReport'
        'documentFormatFiles', 'dataFiles', 'seriesAndClassesContractsInformation']
            section (str): the section to download. For now only supports MANAGEMENT_SECTION or
        RISK_FACTORS

        Returns:
            Optional[str]: section text or None if failed to download
        """
        try:
            if filing[FORM_TYPE] not in cls.file_type_10kq:
                logger.warning(f"Unsupported form type: {filing[FORM_TYPE]}")
                return None

            sec_section: Optional[str] = FILING_DOWNLOAD_LOOKUP[filing[FORM_TYPE]].get(
                section, None
            )
            if sec_section is None:
                logger.warning(f"Unsupported section: {section}")
                return None

            html_text = cls.extractor_api.get_section(
                filing_url=filing[LINK_TO_HTML],
                section=sec_section,
                return_type=DEFAULT_FILING_FORMAT,
            )
            return html.unescape(html_text)
        except Exception as e:
            cik = filing.get(CIK, None)
            company_name = filing.get(COMPANY_NAME, None)
            logger.warning(
                f"Failed to download management section for {company_name=}, ({cik=}): {e}"
            )
            return None

    @classmethod
    def _build_queries_for_filings(
        cls,
        cik: str,
        form_types: List[str],
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> List[Dict]:
        """Build the query string for the SEC Filing API

        Args:
            cik (str): Stock's CIK code
            form_types (List[str]): List of form types to search for
            start_date (Optional[datetime.date]): Start date for the query
            end_date (Optional[datetime.date]): End date for the query. If neither start_date nor
        end_date is provided, the default is to search for the last 90 days

        Returns:
            List[Dict]: A list of query dictionaries. To download the filing via the query, call
        `SecFiling.query_api.get_filings(query)`
        """
        form_types = [form for form in form_types if form in SUPPORTED_TYPE_MAPPING]
        if not form_types:
            return []

        if end_date is None:
            end_date = datetime.datetime.today() + datetime.timedelta(days=1)
        if start_date is None:
            # default to a quarter of data (one filing)
            start_date = end_date - datetime.timedelta(days=90)

        end_date_str = end_date.isoformat()
        start_date_str = start_date.isoformat()

        forms = " OR ".join((f'formType:"{typ}"' for typ in form_types))
        filter_query = f"cik:{cik} AND filedAt:[{start_date_str} TO {end_date_str}] AND ({forms})"

        queries = []
        for i in range(0, 5000, cls.MAX_SEC_QUERY_SIZE):
            queries.append(
                {
                    "query": {"query_string": {"query": filter_query}},
                    "from": str(i),
                    "size": str(i + cls.MAX_SEC_QUERY_SIZE),
                    "sort": [{"filedAt": {"order": "desc"}}],
                }
            )

        return queries

    @classmethod
    def get_filings(
        cls,
        cik: str,
        form_types: List[str],
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> List[Dict]:
        queries = cls._build_queries_for_filings(cik, form_types, start_date, end_date)
        filings = []
        try:
            for query in queries:
                response = cls.query_api.get_filings(query)  # It has built-in retry logic (3 times)
                filings.extend(
                    [
                        filing
                        for filing in response[FILINGS]
                        if filing[FORM_TYPE] in SUPPORTED_TYPE_MAPPING
                    ]
                )

                if len(response[FILINGS]) < cls.MAX_SEC_QUERY_SIZE:
                    # If there are fewer than the max number of allowed
                    # responses for this query, we're done.
                    break
        except Exception as e:
            logger.warning(f"Failed to get the URL to the filings for CIK {cik}: {e}")

        return filings

    @classmethod
    def download_filing_full_content(cls, url: str) -> str:
        text = SecFiling.render_api.get_filing(url)
        return html.unescape(text)