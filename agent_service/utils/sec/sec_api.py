import datetime
import html
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import backoff
from bs4 import BeautifulSoup
from gbi_common_py_utils.utils.ssm import get_param
from sec_api import ExtractorApi, MappingApi, QueryApi, RenderApi  # type: ignore

from agent_service.utils.clickhouse import AsyncClickhouseBase, Clickhouse
from agent_service.utils.date_utils import get_now_utc, parse_date_str_in_utc
from agent_service.utils.sec.constants import (
    CIK,
    COMPANY_NAME,
    CUSIP,
    DEFAULT_FILING_FORMAT,
    DEFAULT_FILINGS_SEARCH_RANGE,
    FILE_10K,
    FILE_10Q,
    FILED_TIMESTAMP,
    FILING_DOWNLOAD_LOOKUP,
    FILINGS,
    FORM_TYPE,
    HTML_PARSER,
    LINK_TO_FILING_DETAILS,
    LINK_TO_HTML,
    LINK_TO_TXT,
    MANAGEMENT_SECTION,
    NASDAQ,
    NYSE,
    RISK_FACTORS,
    SEC_API_KEY_NAME,
    TICKER,
    US,
)
from agent_service.utils.sec.supported_types import SUPPORTED_TYPE_MAPPING
from agent_service.utils.string_utils import get_sections

logger = logging.getLogger(__name__)


def get_file_extension(url: str) -> str:
    _, file_extension = os.path.splitext(url)
    return file_extension


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
    def get_gbi_cik_mapping_from_db(cls, gbi_ids: List[int]) -> Dict[int, Optional[str]]:
        from agent_service.utils.postgres import get_psql

        # `cik` is nullable, null means not supported by SEC
        sql = """
            SELECT gbi_id, cik
            FROM nlp_service.gbi_cik_mapping
            WHERE gbi_id = ANY(%(gbi_ids)s)
        """
        rows = get_psql().generic_read(sql, {"gbi_ids": gbi_ids})
        return {row["gbi_id"]: row["cik"] for row in rows}

    @classmethod
    def get_gbi_cik_mapping_from_api(
        cls, gbi_ids: List[int], insert_to_db: bool = True
    ) -> Dict[int, Optional[str]]:
        if not gbi_ids:
            return {}

        from agent_service.utils.postgres import get_psql

        gbi_id_metadata_map = get_psql().get_sec_metadata_from_gbi(gbi_ids=gbi_ids)
        mapping = {gbi_id: cls.map_gbi_id_to_cik(gbi_id, gbi_id_metadata_map) for gbi_id in gbi_ids}

        if insert_to_db:
            now = get_now_utc()
            rows_to_insert = [
                {"gbi_id": gbi_id, "cik": mapping[gbi_id], "inserted_time": now}
                for gbi_id in gbi_ids
            ]
            get_psql().multi_row_generic_insert_or_update(
                table_name="nlp_service.gbi_cik_mapping",
                rows_to_insert=rows_to_insert,
                conflict="gbi_id",
                columns_to_update=["cik", "inserted_time"],
            )

        return mapping

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


@dataclass(frozen=True)
class SecFilingData:
    db_id: str
    gbi_id: int
    form_type: str
    filed_at: datetime.datetime
    content: str


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
    MAX_DB_QUERY_SIZE = 10

    ################################################################################################
    # Get Filings (metadata)
    ################################################################################################
    @classmethod
    async def get_filings(
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

        filing_to_db_id = await cls._get_db_ids_for_filings(
            gbi_ids, form_types=form_types, start_date=start_date, end_date=end_date
        )  # default to search for last quarter in DB cache

        # Get the CIK mapping for the GBI IDs from DB
        gbi_cik_mapping = SecMapping.get_gbi_cik_mapping_from_db(gbi_ids)

        no_cached_gbi_ids = [gbi_id for gbi_id in gbi_ids if gbi_id not in gbi_cik_mapping]
        gbi_cik_mapping_from_api = SecMapping.get_gbi_cik_mapping_from_api(no_cached_gbi_ids)

        logger.info(
            f"Found {len(gbi_cik_mapping)} out of {len(gbi_ids)} CIKs cached in the database."
        )

        filing_gbi_pairs: List[Tuple[str, int]] = []
        for gbi_id in gbi_ids:
            if gbi_id in gbi_cik_mapping:
                cik = gbi_cik_mapping[gbi_id]
            else:
                cik = gbi_cik_mapping_from_api.get(gbi_id, None)
            if cik is None:
                continue

            try:
                if not start_date and not end_date:
                    filings = []
                    # split up by form type as each will have separate date ranges
                    for form in form_types:
                        # find each document's daterange
                        queries = SecFiling._build_queries_for_filings(cik, form_types=[form])
                        daterange_resp: Optional[Dict] = SecFiling.query_api.get_filings(
                            query=queries.pop()
                        )
                        if (
                            (not daterange_resp)
                            or (FILINGS not in daterange_resp)
                            or (not daterange_resp[FILINGS])
                            or len(daterange_resp[FILINGS]) != 1
                        ):
                            continue  # move onto next form type

                        # filing date of most recent filing
                        latest_filed_at = parse_date_str_in_utc(
                            daterange_resp[FILINGS].pop()[FILED_TIMESTAMP]
                        )

                        # do not include if most recent filing is older than DEFAULT_FILINGS_SEARCH_RANGE
                        if latest_filed_at < get_now_utc() - datetime.timedelta(
                            days=DEFAULT_FILINGS_SEARCH_RANGE
                        ):
                            continue  # move onto next form type

                        # just under one quarter
                        latest_start = latest_filed_at - datetime.timedelta(days=75)

                        # retrieve all documents that match within the recent range
                        queries = SecFiling._build_queries_for_filings(
                            cik,
                            form_types=[form],
                            start_date=latest_start,
                            end_date=latest_filed_at,
                        )
                        for query in queries:
                            document_resp: Optional[Dict] = SecFiling.query_api.get_filings(
                                query=query
                            )
                            if (
                                (not document_resp)
                                or (FILINGS not in document_resp)
                                or (not document_resp[FILINGS])
                            ):
                                continue  # move onto next form type

                            for filing in document_resp[FILINGS]:
                                filings.append(filing)

                            if len(document_resp[FILINGS]) < cls.MAX_SEC_QUERY_SIZE:
                                # If there are fewer than the max number of allowed
                                # responses for this query, we're done.
                                break

                    filings_dict = {json.dumps(filing): gbi_id for filing in filings}
                    filing_gbi_pairs.extend(list(filings_dict.items()))
                else:
                    queries = SecFiling._build_queries_for_filings(
                        cik, form_types=form_types, start_date=start_date, end_date=end_date
                    )
                    for query in queries:
                        resp: Optional[Dict] = SecFiling.query_api.get_filings(query=query)
                        if (not resp) or (FILINGS not in resp):
                            continue
                        elif not resp[FILINGS]:
                            break

                        filing_gbi_pairs.extend(
                            [(json.dumps(filing), gbi_id) for filing in resp[FILINGS]]
                        )

                        if len(resp[FILINGS]) < cls.MAX_SEC_QUERY_SIZE:
                            # If there are fewer than the max number of allowed
                            # responses for this query, we're done.
                            break
                time.sleep(0.5)  # avoid rate limit
            except Exception as e:
                logger.exception(f"Failed to get filings for {gbi_id=}: {e}")
                time.sleep(10)

        logger.info(
            f"Found {len(filing_gbi_pairs)} filings for {len(gbi_ids)} stocks."
            f"Found {len(filing_to_db_id)} filings cached in the database."
        )

        return filing_gbi_pairs, filing_to_db_id

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
            start_date (datetime.date): Start date for the query
            end_date (datetime.date): End date for the query.
        If neither start_date nor end_date is provided, it will find the most recent filing (1)

        Returns:
            List[Dict]: A list of query dictionaries. To download the filing via the query, call
        `SecFiling.query_api.get_filings(query)`
        """
        form_types = [form for form in form_types if form in SUPPORTED_TYPE_MAPPING]
        if not form_types:
            return []

        filed_at = ""
        if start_date and end_date:
            end_date_str = end_date.isoformat()
            start_date_str = start_date.isoformat()
            filed_at = f"AND filedAt:[{start_date_str} TO {end_date_str}]"

        forms = " OR ".join((f'formType:"{typ}"' for typ in form_types))
        filter_query = f"cik:{cik} {filed_at} AND ({forms})"

        queries = []
        if filed_at == "":
            # If no date range is provided, just find most recent filing
            queries.append(
                {
                    "query": {"query_string": {"query": filter_query}},
                    "from": "0",
                    "size": "1",
                    "sort": [{"filedAt": {"order": "desc"}}],
                }
            )
            return queries

        for i in range(0, 5000, cls.MAX_SEC_QUERY_SIZE):
            queries.append(
                {
                    "query": {"query_string": {"query": filter_query}},
                    "from": str(i),
                    "size": str(cls.MAX_SEC_QUERY_SIZE),
                    "sort": [{"filedAt": {"order": "desc"}}],
                }
            )

        return queries

    @classmethod
    async def _get_db_ids_for_filings(
        cls,
        gbi_ids: List[int],
        form_types: List[str],
        start_date: Optional[datetime.date] = None,  # inclusive
        end_date: Optional[datetime.date] = None,  # inclusive
    ) -> Dict[str, str]:
        """
        Given a list of SEC filings, return a list of database table IDs for the available filings.
        Default date range will be the last 100 days.
        """
        if end_date is None:
            end_date = datetime.date.today() + datetime.timedelta(days=1)
        if start_date is None:
            # default to a quarter of data (one filing)
            start_date = end_date - datetime.timedelta(days=100)

        sql = """
            SELECT DISTINCT ON (formType, gbi_id, filedAt)
                id::TEXT AS id, filing
            FROM sec.sec_filings
            WHERE gbi_id IN %(gbi_ids)s AND formType IN %(form_types)s
                AND filedAt >= %(start_date)s AND filedAt <= %(end_date)s
        """
        ch = Clickhouse()
        result = await ch.generic_read(
            sql,
            params={
                "gbi_ids": gbi_ids,
                "form_types": form_types,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        return {row["filing"]: row["id"] for row in result}

    ################################################################################################
    # Get sections for 10K/10Q filings
    ################################################################################################
    @classmethod
    async def get_concat_10k_10q_sections_from_db(
        cls, db_id_to_text_id: Dict[str, str]
    ) -> Dict[str, str]:
        if not db_id_to_text_id:
            return {}

        sql = """
            SELECT id::TEXT AS id, content, riskFactors, managementSection
            FROM sec.sec_filings
            WHERE formType in ('10-K', '10-Q') AND id IN %(db_ids)s
        """
        ch = Clickhouse()

        output = {}

        db_ids = list(db_id_to_text_id.keys())
        for idx in range(0, len(db_ids), cls.MAX_DB_QUERY_SIZE):
            batch_db_ids = db_ids[idx : idx + cls.MAX_DB_QUERY_SIZE]
            result = await ch.generic_read(sql, params={"db_ids": batch_db_ids})

            for row in result:
                risk_factor_section = row["riskFactors"]
                management_section = row["managementSection"]
                if risk_factor_section or management_section:
                    text = (
                        f"Management Section:\n\n{management_section}\n\n"
                        f"Risk Factors Section:\n\n{risk_factor_section}"
                    )
                else:
                    text = BeautifulSoup(row["content"], HTML_PARSER).getText()

                filing_id = db_id_to_text_id[row["id"]]
                output[filing_id] = text

        return output

    @classmethod
    async def get_concat_10k_10q_sections_from_db_by_filing_jsons(
        cls, filing_jsons: List[str]
    ) -> Dict[str, Tuple[str, str]]:
        if not filing_jsons:
            return {}

        sql = """
            SELECT id::TEXT AS id, filing, content, riskFactors, managementSection
            FROM sec.sec_filings
            WHERE formType in ('10-K', '10-Q') AND filing IN %(filing_jsons)s
        """
        ch = Clickhouse()

        output = {}
        for idx in range(0, len(filing_jsons), cls.MAX_DB_QUERY_SIZE):
            batch_filing_jsons = filing_jsons[idx : idx + cls.MAX_DB_QUERY_SIZE]
            result = await ch.generic_read(sql, params={"filing_jsons": batch_filing_jsons})
            for row in result:
                risk_factor_section = row["riskFactors"]
                management_section = row["managementSection"]

                if risk_factor_section or management_section:
                    text = (
                        f"Management Section:\n\n{management_section}\n\n"
                        f"Risk Factors Section:\n\n{risk_factor_section}"
                    )
                else:
                    text = BeautifulSoup(row["content"], HTML_PARSER).getText()

                output[row["filing"]] = (row["id"], text)

        return output

    @classmethod
    async def get_concat_10k_10q_sections_from_api(
        cls, filing_gbi_pairs: List[Tuple[str, int]], insert_to_db: bool = True
    ) -> Dict[str, str]:
        if not filing_gbi_pairs:
            return {}

        ch = Clickhouse()

        output = {}
        for filing_info_str, gbi_id in filing_gbi_pairs:
            filing_info = json.loads(filing_info_str)

            try:
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

                # LINK_TO_HTML is ok for extracting sections, but LINK_TO_FILING_DETAILS is needed for full content
                # Examples:
                # LINK_TO_HTML: https://www.sec.gov/Archives/edgar/data/320193/000032019324000069/0000320193-24-000069-index.htm  # noqa
                # LINK_TO_FILING_DETAILS: https://www.sec.gov/Archives/edgar/data/320193/000032019324000069/aapl-20240330.htm  # noqa
                # For some older SEC filings (pre-2000), LINK_TO_FILING_DETAILS returns a directory
                # and not the actual text, so we use LINK_TO_TXT when LINK_TO_FILING_DETAILS has no extension
                # Examples:
                # LINK_TO_FILING_DETAILS: https://www.sec.gov/Archives/edgar/data/320193/  # noqa
                # LINK_TO_TXT: https://www.sec.gov/Archives/edgar/data/320193/0000912057-97-019277.txt  # noqa

                extension: str = get_file_extension(filing_info[LINK_TO_FILING_DETAILS])
                # no extension == directory
                if len(extension) == 0:
                    full_content = SecFiling.render_api.get_filing(url=filing_info[LINK_TO_TXT])
                else:
                    full_content = SecFiling.render_api.get_filing(
                        url=filing_info[LINK_TO_FILING_DETAILS]
                    )

                parsed_full_content = BeautifulSoup(html.unescape(full_content), HTML_PARSER)
                processed_full_content = parsed_full_content.getText()
                time.sleep(0.25)

                if management_section or risk_factor_section:
                    # if there's at least 1 non-empty section, use the concatenated text
                    output[filing_info_str] = text
                else:
                    output[filing_info_str] = processed_full_content

                if insert_to_db:
                    try:
                        await ch.multi_row_insert(
                            table_name="sec.sec_filings",
                            rows=[
                                {
                                    "gbi_id": gbi_id,
                                    CIK: filing_info[CIK],
                                    FORM_TYPE: filing_info[FORM_TYPE],  # '10-Q' or '10-K
                                    FILED_TIMESTAMP: parse_date_str_in_utc(
                                        filing_info[FILED_TIMESTAMP]
                                    ),
                                    "filing": filing_info_str,
                                    "content": processed_full_content,
                                    MANAGEMENT_SECTION: management_section,
                                    RISK_FACTORS: risk_factor_section,
                                }
                            ],
                        )
                    except Exception as e:
                        # FIXME: there seems to be some size-limitation issue (2MB), right now just
                        # log the error and skip the insertion
                        logger.exception(f"Failed to insert filing content for {gbi_id=}: {e}")
            except Exception as e:
                logger.exception(f"Failed to get 10K/10Q filing sections for {gbi_id=}: {e}")
                time.sleep(10)

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
            processed_html_text = html.unescape(html_text)
            time.sleep(0.25)
            return processed_html_text
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
    async def get_filings_content_from_db(cls, db_id_to_text_id: Dict[str, str]) -> Dict[str, str]:
        if not db_id_to_text_id:
            return {}

        sql = """
            SELECT id::TEXT AS id, content
            FROM sec.sec_filings
            WHERE id IN %(db_ids)s
        """
        ch = Clickhouse()

        output = {}
        db_ids = list(db_id_to_text_id.keys())
        for idx in range(0, len(db_ids), cls.MAX_DB_QUERY_SIZE):
            batch_db_ids = db_ids[idx : idx + cls.MAX_DB_QUERY_SIZE]
            result = await ch.generic_read(sql, params={"db_ids": batch_db_ids})

            for row in result:
                filing_id = db_id_to_text_id[row["id"]]
                output[filing_id] = BeautifulSoup(row["content"], HTML_PARSER).getText()

        return output

    @classmethod
    async def get_filings_content_from_db_by_filing_jsons(
        cls, filing_jsons: List[str]
    ) -> Dict[str, Tuple[str, str]]:
        if not filing_jsons:
            return {}

        sql = """
            SELECT id::TEXT AS id, filing, content
            FROM sec.sec_filings
            WHERE filing IN %(filing_jsons)s
        """
        ch = Clickhouse()

        output = {}
        for idx in range(0, len(filing_jsons), cls.MAX_DB_QUERY_SIZE):
            batch_filing_jsons = filing_jsons[idx : idx + cls.MAX_DB_QUERY_SIZE]
            result = await ch.generic_read(sql, params={"filing_jsons": batch_filing_jsons})

            for row in result:
                output[row["filing"]] = (
                    row["id"],
                    BeautifulSoup(row["content"], HTML_PARSER).getText(),
                )

        return output

    @classmethod
    async def get_filing_data_async(cls, db_ids: List[str]) -> Dict[str, SecFilingData]:
        sql = """
            SELECT id::TEXT AS db_id, riskFactors, managementSection, gbi_id,
                   formType AS form_type, filedAt AS filed_at, content
            FROM sec.sec_filings
            WHERE id IN %(db_ids)s
        """
        ch = AsyncClickhouseBase()
        result = await ch.generic_read(sql, params={"db_ids": db_ids})

        output = {}
        for row in result:
            filing_id = row["db_id"]
            management_section = row.pop("managementSection")
            risk_factor_section = row.pop("riskFactors")
            row["content"] = (
                (
                    f"Management Section:\n\n{management_section}\n\n"
                    f"Risk Factors Section:\n\n{risk_factor_section}"
                )
                if management_section and risk_factor_section
                else BeautifulSoup(row["content"], HTML_PARSER).getText()
            )
            output[filing_id] = SecFilingData(**row)

        return output

    @classmethod
    async def get_filing_data_by_type_date_async(
        cls, gbi_id: int, filing_type: str, date: datetime.date
    ) -> Optional[SecFilingData]:
        sql = """
            SELECT id::TEXT AS db_id, riskFactors, managementSection, gbi_id,
                   formType AS form_type, filedAt AS filed_at, content
            FROM sec.sec_filings
            WHERE gbi_id = %(gbi_id)s AND filedAt::DATE = %(filed_at)s
                  AND formType = %(filing_type)s
        """
        params = {"gbi_id": gbi_id, "filed_at": date, "filing_type": filing_type}
        ch = AsyncClickhouseBase()
        result = await ch.generic_read(
            sql,
            params=params,
        )

        if len(result) == 0:
            return None
        row = result[0]
        management_section = row.pop("managementSection")
        risk_factor_section = row.pop("riskFactors")
        row["content"] = (
            (
                f"Management Section:\n\n{management_section}\n\n"
                f"Risk Factors Section:\n\n{risk_factor_section}"
            )
            if management_section and risk_factor_section
            else BeautifulSoup(row["content"], HTML_PARSER).getText()
        )
        return SecFilingData(**row)

    @classmethod
    async def get_filings_content_from_api(
        cls, filing_gbi_pairs: List[Tuple[str, int]], insert_to_db: bool = True
    ) -> Dict[str, str]:
        if not filing_gbi_pairs:
            return {}

        ch = Clickhouse()

        output = {}
        for filing_info_str, gbi_id in filing_gbi_pairs:
            filing_info = json.loads(filing_info_str)

            try:
                extension: str = get_file_extension(filing_info[LINK_TO_FILING_DETAILS])
                # no extension == directory
                if len(extension) == 0:
                    full_content = SecFiling.render_api.get_filing(url=filing_info[LINK_TO_TXT])
                else:
                    full_content = SecFiling.render_api.get_filing(
                        url=filing_info[LINK_TO_FILING_DETAILS]
                    )

                parsed_full_content = BeautifulSoup(html.unescape(full_content), HTML_PARSER)
                processed_full_content = parsed_full_content.getText()

                output[filing_info_str] = processed_full_content

                if insert_to_db:
                    try:
                        await ch.multi_row_insert(
                            table_name="sec.sec_filings",
                            rows=[
                                {
                                    "gbi_id": gbi_id,
                                    CIK: filing_info[CIK],
                                    FORM_TYPE: filing_info[FORM_TYPE],
                                    FILED_TIMESTAMP: parse_date_str_in_utc(
                                        filing_info[FILED_TIMESTAMP]
                                    ),
                                    "filing": filing_info_str,
                                    "content": processed_full_content,
                                }
                            ],
                        )
                    except Exception as e:
                        # FIXME: there seems to be some size-limitation issue (2MB), right now just
                        # log the error and skip the insertion
                        logger.exception(f"Failed to insert filing content for {gbi_id=}: {e}")

                time.sleep(0.5)  # avoid rate limit
            except Exception as e:
                logger.exception(f"Failed to get filing content for {gbi_id=}: {e}")
                time.sleep(10)

        return output

    ################################################################################################
    # Convert 10k/q content into smaller sections
    ################################################################################################
    @classmethod
    def split_10k_10q_into_smaller_sections(cls, filing_text: str) -> Dict[str, str]:
        two_sections = filing_text.split("\n\nRisk Factors Section:\n\n")
        if len(two_sections) != 2:
            # if the report doesn't have the risk factors section
            return {}

        management_section = two_sections[0]
        management_section = management_section[len("Management Section:\n\n") :]

        risk_factors_section = two_sections[1]
        # FIXME: the value will be overwritten when there are same headers
        smaller_sections = get_sections(management_section)
        smaller_sections.update(get_sections(risk_factors_section))

        return smaller_sections
