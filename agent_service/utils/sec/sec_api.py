import datetime
import html
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import backoff
from gbi_common_py_utils.utils.ssm import get_param
from sec_api import ExtractorApi, MappingApi, QueryApi, RenderApi  # type: ignore

from agent_service.utils.clickhouse import Clickhouse
from agent_service.utils.date_utils import parse_date_str_in_utc
from agent_service.utils.sec.constants import (
    CIK,
    COMPANY_NAME,
    CUSIP,
    DEFAULT_FILING_FORMAT,
    FILE_10K,
    FILE_10Q,
    FILED_TIMESTAMP,
    FILING_DOWNLOAD_LOOKUP,
    FILINGS,
    FORM_TYPE,
    LINK_TO_FILING_DETAILS,
    LINK_TO_HTML,
    MANAGEMENT_SECTION,
    NASDAQ,
    NYSE,
    RISK_FACTORS,
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

    NOTE: Please keep these methods SYNCHRONOUS - this is being used in an
    async context (agent) and we are relying on this non parallel nature to
    help us implicitly rate limit ourselves by doing these calls sequentially
    in lieu of a real rate limiting solution.
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

    NOTE: Please keep these methods SYNCHRONOUS - this is being used in an
    async context (agent) and we are relying on this non parallel nature to
    help us implicitly rate limit ourselves by doing these calls sequentially
    in lieu of a real rate limiting solution.
    """

    api_key = get_param(SEC_API_KEY_NAME)
    query_api = QueryApi(api_key)  # use it to get metadata (e.g. URL) of the filings
    extractor_api = ExtractorApi(api_key)  # use it to extract the sections from the filings
    render_api = RenderApi(api_key)  # use it to download the full content of filings

    file_type_10kq = {FILE_10K, FILE_10Q}

    MAX_SEC_QUERY_SIZE = 50

    ################################################################################################
    # Get Filings (metadata)
    ################################################################################################
    @classmethod
    def get_filings(
        cls,
        gbi_ids: List[int],
        form_types: List[str],
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> Tuple[List[Tuple[str, int]], Dict[str, str]]:
        """
        Given a list of GBI IDs, and a date range, return 2 things:
        1. A list of pairs (10K/Q filing string, GBI ID)
        2. A dictionary mapping the available filings to their database table IDs

        Use this method to know what filings are available in our own DB and use IDs to fetch them later,
        and what are needed to fetch through API
        """
        form_types = [form for form in form_types if form in SUPPORTED_TYPE_MAPPING]
        if not form_types:
            raise Exception("Couldn't find any supported SEC filing types in the request.")

        if end_date is None:
            end_date = datetime.date.today() + datetime.timedelta(days=1)
        if start_date is None:
            # default to a quarter of data (one filing)
            start_date = end_date - datetime.timedelta(days=90)

        filing_to_db_id = cls._get_db_ids_for_filings(
            gbi_ids, form_types=form_types, start_date=start_date, end_date=end_date
        )

        from agent_service.utils.postgres import get_psql

        gbi_id_metadata_map = get_psql().get_sec_metadata_from_gbi(gbi_ids=gbi_ids)
        filing_gbi_pairs: List[Tuple[str, int]] = []
        for gbi_id in gbi_ids:
            cik = SecMapping.map_gbi_id_to_cik(gbi_id, gbi_id_metadata_map)
            if cik is None:
                continue

            queries = SecFiling._build_queries_for_filings(
                cik, form_types=form_types, start_date=start_date, end_date=end_date
            )
            for query in queries:
                resp: Optional[Dict] = SecFiling.query_api.get_filings(query=query)
                if (not resp) or (FILINGS not in resp) or (not resp[FILINGS]):
                    continue

                filing_gbi_pairs.extend([(json.dumps(filing), gbi_id) for filing in resp[FILINGS]])

                if len(resp[FILINGS]) < cls.MAX_SEC_QUERY_SIZE:
                    # If there are fewer than the max number of allowed
                    # responses for this query, we're done.
                    break

        logger.info(
            f"Found {len(filing_gbi_pairs)} filings for {len(gbi_ids)} stocks "
            f"between {start_date} and {end_date}. "
            f"Found {len(filing_to_db_id)} filings cached in the database."
        )

        return filing_gbi_pairs, filing_to_db_id

    @classmethod
    def _build_queries_for_filings(
        cls,
        cik: str,
        form_types: List[str],
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> List[Dict]:
        """Build the query string for the SEC Filing API

        Args:
            cik (str): Stock's CIK code
            form_types (List[str]): List of form types to search for
            start_date (datetime.date): Start date for the query
            end_date (datetime.date): End date for the query. If neither start_date nor
        end_date is provided, the default is to search for the last 90 days

        Returns:
            List[Dict]: A list of query dictionaries. To download the filing via the query, call
        `SecFiling.query_api.get_filings(query)`
        """
        form_types = [form for form in form_types if form in SUPPORTED_TYPE_MAPPING]
        if not form_types:
            return []

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
    def _get_db_ids_for_filings(
        cls,
        gbi_ids: List[int],
        form_types: List[str],
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> Dict[str, str]:
        """
        Given a list of SEC filings, return a list of database table IDs for the available filings
        """

        sql = """
            SELECT DISTINCT ON (formType, gbi_id, filedAt)
                id::TEXT, filing
            FROM sec.sec_filings
            WHERE gbi_id IN %(gbi_ids)s AND formType IN %(form_types)s
                AND filedAt >= %(start_date)s AND filedAt <= %(end_date)s
        """
        ch = Clickhouse()
        result = ch.clickhouse_client.query(
            sql,
            parameters={
                "gbi_ids": gbi_ids,
                "form_types": form_types,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        return {tup[1]: tup[0] for tup in result.result_rows}

    ################################################################################################
    # Get sections for 10K/10Q filings
    ################################################################################################
    @classmethod
    def get_concat_10k_10q_sections_from_db(
        cls, db_id_to_text_id: Dict[str, str]
    ) -> Dict[str, str]:
        sql = """
            SELECT id::TEXT, riskFactors, managementSection
            FROM sec.sec_filings
            WHERE formType in ('10-K', '10-Q') AND id IN %(db_ids)s
        """
        ch = Clickhouse()
        result = ch.clickhouse_client.query(
            sql, parameters={"db_ids": list(db_id_to_text_id.keys())}
        )

        output = {}
        for tup in result.result_rows:
            risk_factor_section = tup[1]
            management_section = tup[2]
            text = (
                f"Management Section:\n\n{management_section}\n\n"
                f"Risk Factors Section:\n\n{risk_factor_section}"
            )

            filing_id = db_id_to_text_id[tup[0]]
            output[filing_id] = text

        return output

    @classmethod
    def get_concat_10k_10q_sections_from_api(
        cls, filing_gbi_pairs: List[Tuple[str, int]], insert_to_db: bool = True
    ) -> Dict[str, str]:
        output = {}
        records_to_upload_to_db: List[Dict] = []
        for filing_info_str, gbi_id in filing_gbi_pairs:
            filing_info = json.loads(filing_info_str)

            # NOTE that these downloaded sections are processed, not the raw data
            management_section = SecFiling._download_10k_10q_section(
                filing_info, section=MANAGEMENT_SECTION
            )
            risk_factor_section = SecFiling._download_10k_10q_section(
                filing_info, section=RISK_FACTORS
            )
            text = (
                f"Management Section:\n\n{management_section}\n\n"
                f"Risk Factors Section:\n\n{risk_factor_section}"
            )
            output[filing_info_str] = text

            # LINK_TO_HTML is ok for extracting sections, but LINK_TO_FILING_DETAILS is needed for full content
            # See examples here:
            # LINK_TO_HTML: https://www.sec.gov/Archives/edgar/data/320193/000032019324000069/0000320193-24-000069-index.htm  # noqa
            # LINK_TO_FILING_DETAILS: https://www.sec.gov/Archives/edgar/data/320193/000032019324000069/aapl-20240330.htm  # noqa

            full_content = SecFiling.render_api.get_filing(url=filing_info[LINK_TO_FILING_DETAILS])
            processed_full_content = html.unescape(full_content)

            records_to_upload_to_db.append(
                {
                    "gbi_id": gbi_id,
                    CIK: filing_info[CIK],
                    FORM_TYPE: filing_info[FORM_TYPE],  # '10-Q' or '10-K
                    FILED_TIMESTAMP: parse_date_str_in_utc(filing_info[FILED_TIMESTAMP]),
                    "filing": filing_info_str,
                    "content": processed_full_content,
                    MANAGEMENT_SECTION: management_section,
                    RISK_FACTORS: risk_factor_section,
                }
            )

        if insert_to_db and records_to_upload_to_db:
            ch = Clickhouse()
            ch.multi_row_insert(table_name="sec.sec_filings", rows=records_to_upload_to_db)

        return output

    @classmethod
    def _download_10k_10q_section(cls, filing: Dict, section: str) -> Optional[str]:
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

    ################################################################################################
    # Get full content of filings
    ################################################################################################
    @classmethod
    def get_filings_content_from_db(cls, db_id_to_text_id: Dict[str, str]) -> Dict[str, str]:
        sql = """
            SELECT id::TEXT, content
            FROM sec.sec_filings
            WHERE id IN %(db_ids)s
        """
        ch = Clickhouse()
        result = ch.clickhouse_client.query(
            sql, parameters={"db_ids": list(db_id_to_text_id.keys())}
        )

        output = {}
        for tup in result.result_rows:
            filing_id = db_id_to_text_id[tup[0]]
            output[filing_id] = tup[1]

        return output

    @classmethod
    def get_filings_content_from_api(
        cls, filing_gbi_pairs: List[Tuple[str, int]], insert_to_db: bool = True
    ) -> Dict[str, str]:
        output = {}
        records_to_upload_to_db: List[Dict] = []
        for filing_info_str, gbi_id in filing_gbi_pairs:
            filing_info = json.loads(filing_info_str)

            text = SecFiling.render_api.get_filing(url=filing_info[LINK_TO_FILING_DETAILS])
            processed_text = html.unescape(text)

            output[filing_info_str] = processed_text

            records_to_upload_to_db.append(
                {
                    "gbi_id": gbi_id,
                    CIK: filing_info[CIK],
                    FORM_TYPE: filing_info[FORM_TYPE],
                    FILED_TIMESTAMP: parse_date_str_in_utc(filing_info[FILED_TIMESTAMP]),
                    "filing": filing_info_str,
                    "content": processed_text,
                }
            )

        if insert_to_db and records_to_upload_to_db:
            ch = Clickhouse()
            ch.multi_row_insert(table_name="sec.sec_filings", rows=records_to_upload_to_db)

        return output
