import asyncio
import io
import json
import logging
from collections import defaultdict
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry, TableColumnType
from agent_service.io_types.table import Table
from agent_service.io_types.text import KPIText, TextCitation
from agent_service.tools.product_comparison.brightdata_websearch import (
    brd_request,
    brd_websearch,
)
from agent_service.tools.product_comparison.constants import CHUNK_SIZE
from agent_service.tools.product_comparison.helpers import update_dataframe
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed

WEB_SCRAPE_PRODUCT_NAME_MAIN_PROMPT = (
    "Given part of a product page '{html}', extract the most relevant {product_name} which is often the latest "
    "product from {company_name} from the page."
    "Be SURE that the product you are fetching is from {company_name} and is a {product_name}"
    "I ONLY want a SINGLE product name to be returned, no extra details."
    "If the most relevant product name is not found, or if the found product isn't a product, do not output anything."
    "If the most relevant product is not comparable or related to the latest {product_name} from {main_stock}, "
    "do not output anything. Outputting something unrelated is MUCH WORSE than outputting nothing!"
    "Do not use additional formatting such as newlines or added spaces for human readability."
    "You should only output the SINGLE product name, even if products are related, no justification"
)

WEB_SCRAPE_PRODUCT_NAME_SYS_PROMPT = (
    "You are a financial analyst that is doing research on company products."
    "You have been tasked with finding the name of the most relevant released {product_name} from {company_name}."
    "You will be provided with part of the company's product page as well as the type of product looked for."
    "Your task is to locate the name of the most relevant released {product_name} which has already been released."
    "It is important that you ONLY output a single product name. "
    "It is VERY IMPORTANT that the product you are fetching is from {company_name} and is a {product_name}, "
    "if it is not, it is better that you output NOTHING"
    "As well, do not output anything if the product is not a {product_name}"
)

WEB_SCRAPE_PRODUCT_MAIN_PROMPT = (
    "Given part of a company's product page {html} from {url} with title {title} and a dictionary of keys which need "
    "values to be filled in {product_specs}, find in the values of dictionary for all keys except for stock_id. "
    "Output the exact same input dictionary but with values filled in. "
    "Be as detailed as you can with numerical values as well as textual context with the information you're provided"
    "Your value should be a dict with the validated information 'value', 'url' source, website 'title',"
    "as well as the 'justification'. "
    "If a value already has an assigned URL or was not found, don’t touch the existing value. "
    "DO NOT add any more values into the dictionary"
    "No other details, justification or formatting such as \\n, "
    "``` or python should be returned, I only want the dictionary."
)

WEB_SCRAPE_PRODUCT_SYS_PROMPT = (
    "You are a financial analyst that is doing research on company products."
    "You have been tasked with finding and outputting product information for a list of products."
    "You will be provided with part of the company's product page."
    "You will be provided with the product name, company name and "
    "product specifications that your colleague has compiled."
    "Your task is to find and fill out the product information, be detailed when you have the available information "
    "so the product information is easy to understand and compare."
    "You must also include the url source in your output"
)

FINAL_VALIDATION_MAIN_PROMPT = (
    "You have been given a list of validation notes from your colleage's {validations}."
    "The same information may differ between the notes."
    "You must go through the justification and determine the correct information."
    "Output the validated information as a dictionary."
    "If {validations} is empty, you should output an empty dictionary."
    "Do not output information that is not present in the notes."
    "But if the information is present in the notes, you should output it and be as detailed as you can."
    "Make sure to include the url of the page that the information was found on."
    "Your key should be a dict with the validated information 'value', 'url' source, website 'title',"
    "as well as the 'justification'."
    "Do not mention 'n/a' in the justification. You should just say what information was found."
    "You should not mention that the previous value was incorrect."
    "If information was not found, don't say anything."
    "As well, don't justify the names like company name, product name."
    "Make sure this is the case for every key in the dict including stock_id dict"
    "DO NOT add any more values into the dictionary"
)

FINAL_VALIDATION_SYS_PROMPT = (
    "You are a financial analyst that is doing research on company products."
    "You have been tasked with validating product information for a list of products."
    "You will be provided with a list of validation notes from your colleages."
    "These will all contain url information as well."
    "Your task is to validate the product information."
    "Do not add comments or any other text to the dict."
    "Make sure to use "
    " for strings and not ''."
    "Make sure that stock_id is the int gbi_id of the stock."
)

WEB_SCRAPE_PRODUCT_NAME_MAIN_PROMPT_OBJ = Prompt(
    name="WEB_SCRAPE_PRODUCT_NAME_MAIN_PROMPT",
    template=WEB_SCRAPE_PRODUCT_NAME_MAIN_PROMPT,
)

WEB_SCRAPE_PRODUCT_NAME_SYS_PROMPT_OBJ = Prompt(
    name="WEB_SCRAPE_PRODUCT_NAME_SYS_PROMPT",
    template=WEB_SCRAPE_PRODUCT_NAME_SYS_PROMPT,
)

WEB_SCRAPE_PRODUCT_MAIN_PROMPT_OBJ = Prompt(
    name="WEB_SCRAPE_PRODUCT_MAIN_PROMPT",
    template=WEB_SCRAPE_PRODUCT_MAIN_PROMPT,
)

WEB_SCRAPE_PRODUCT_SYS_PROMPT_OBJ = Prompt(
    name="WEB_SCRAPE_PRODUCT_SYS_PROMPT",
    template=WEB_SCRAPE_PRODUCT_SYS_PROMPT,
)


FINAL_VALIDATION_MAIN_PROMPT_OBJ = Prompt(
    name="FINAL_VALIDATION_MAIN_PROMPT",
    template=FINAL_VALIDATION_MAIN_PROMPT,
)

FINAL_VALIDATION_SYS_PROMPT_OBJ = Prompt(
    name="FINAL_VALIDATION_SYS_PROMPT",
    template=FINAL_VALIDATION_SYS_PROMPT,
)

logger = logging.getLogger(__name__)


def get_specifications(company_name: str, product: str, important_specs: List[str]) -> List[str]:
    result = set()
    result.update(brd_websearch(f"{product} release date", 2))
    result.update(brd_websearch(f"{product} specs", 2))
    for spec in important_specs:
        result.update(brd_websearch(f"{company_name} {product} {spec}", 1))
    return list(result)


class WebScraper:
    def __init__(self, context: PlanRunContext) -> None:
        gpt_context = create_gpt_context(
            GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
        )
        self.llm = GPT(context=gpt_context, model=GPT4_O)

    async def validate_product_info(
        self, product_table: Table, important_specs: List[str]
    ) -> Table:
        df = product_table.to_df()
        company_urls = {
            stock_id: (get_specifications(stock_id.company_name, product, important_specs))
            for stock_id, product in zip(df["stock_id"], df["product_name"])
        }

        tasks = []

        for i in range(len(df)):
            row = df.iloc[i]
            stock = row.stock_id
            stock_urls = company_urls.get(stock, [])
            tasks.append(self.explore_page(stock_urls, row.to_dict()))

        results = await gather_with_concurrency(tasks)
        row_descs: List[HistoryEntry] = []
        for i in range(len(df)):
            url_justifications = [
                (
                    results[i].get(key, {}).get("title"),
                    results[i].get(key, {}).get("url"),
                    results[i].get(key, {}).get("justification"),
                )
                for key in results[i].keys()
            ]
            urls_to_justifications = defaultdict(list)

            for title, url, justification in url_justifications:
                if url is not None and justification:
                    urls_to_justifications[(title, url)].append(justification)

            row_descs.append(
                HistoryEntry(
                    citations=[
                        TextCitation(
                            source_text=KPIText(
                                val=str(title), explanation=" ".join(justifications), url=str(url)
                            )
                        )
                        for (title, url), justifications in urls_to_justifications.items()
                        if url is not None
                    ],
                )
            )

        # If x is found in "results", replace the value in the dataframe
        for i in range(len(df)):
            for j, col in enumerate(product_table.columns):
                if col.metadata.label != "stock_id":
                    df.iat[i, j] = results[i].get(col.metadata.label, {}).get("value", "n/a")

        # Editing column metadata to adapt to column changes we just made
        metadata = [product_table.columns[k].metadata for k in range(len(product_table.columns))]
        for i, column in enumerate(metadata):
            if column.col_type == TableColumnType.STOCK:
                continue
            else:
                col_label = str(column.label)
                col_values = df[col_label].tolist()
                metadata[i] = update_dataframe(col_label, col_values, df)

        product_table = Table.from_df_and_cols(metadata, df)
        product_table.history = row_descs
        return product_table

    async def explore_page(self, urls: List[str], product_specs: Dict) -> Dict:
        tasks = []
        for url in urls:
            try:
                response = brd_request(url)
                content_type = response.headers.get("Content-Type")
                if "application/pdf" in content_type:
                    pdf_binary_data = response.content
                    pdf_file = io.BytesIO(pdf_binary_data)
                    pdf_reader = PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                elif "text/html" in content_type:
                    html_content = response.text
                    soup = BeautifulSoup(html_content, "html.parser")
                    text = soup.get_text()
                    title = soup.title.string if soup.title else url
                else:
                    logger.info(f"Unsupported content type: {content_type}")
                    continue
            except requests.exceptions.HTTPError as e:
                logger.info(
                    f"Failed to retrieve {url}. HTTPError: {e.response.status_code} - {e.response.reason}"
                )
                continue
            except requests.exceptions.RequestException as e:
                if e.response:
                    logger.info(f"URLError: {e.response.reason}")
                continue
            except TimeoutError as e:
                logger.info(f"TimeoutError: {e}")
                continue
            except Exception as e:
                logger.info(f"Failed to retrieve {url}. Error: {e}")
                continue

            chunks = []

            # Splitting text into chunks with max length of 20000 characters, but not breaking up paragraphs
            while len(text) > CHUNK_SIZE:
                chunk = text[:CHUNK_SIZE]
                last_newline = chunk.rfind("\n")
                chunks.append(chunk[:last_newline])
                text = text[last_newline:]
            chunks.append(text)

            for chunk in chunks:
                tasks.append(
                    self.llm.do_chat_w_sys_prompt(
                        WEB_SCRAPE_PRODUCT_SYS_PROMPT_OBJ.format(),
                        WEB_SCRAPE_PRODUCT_MAIN_PROMPT_OBJ.format(
                            html=chunk, product_specs=product_specs, url=url, title=title
                        ),
                    )
                )

        first_pass = await gather_with_concurrency(tasks)
        final_result = await self.llm.do_chat_w_sys_prompt(
            FINAL_VALIDATION_MAIN_PROMPT_OBJ.format(validations=first_pass),
            FINAL_VALIDATION_SYS_PROMPT_OBJ.format(),
        )
        final_result = clean_to_json_if_needed(final_result)
        validated = json.loads(final_result)

        validated["stock_id"] = {"value": product_specs["stock_id"], "url": ""}
        validated["product_name"] = {"value": product_specs["product_name"], "url": ""}
        return validated

    async def get_product_name(self, company_name: str, product: str, main_stock: str) -> str:
        if not company_name:
            return "n/a"

        urls = brd_websearch(f"latest released {company_name} {product}", 2)
        tasks = []

        for url in urls:
            try:
                response = brd_request(url)
                content_type = response.headers.get("Content-Type")
                if "application/pdf" in content_type:
                    pdf_binary_data = response.content
                    pdf_file = io.BytesIO(pdf_binary_data)
                    pdf_reader = PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                elif "text/html" in content_type:
                    html_content = response.text
                    soup = BeautifulSoup(html_content, "html.parser")
                    text = soup.get_text()
                else:
                    logger.info(f"Unsupported content type: {content_type}")
                    continue
            except requests.exceptions.HTTPError as e:
                logger.info(
                    f"Failed to retrieve {url}. HTTPError: {e.response.status_code} - {e.response.reason}"
                )
                continue
            except requests.exceptions.RequestException as e:
                if e.response:
                    logger.info(f"URLError: {e.response.reason}")
                continue
            except TimeoutError as e:
                logger.info(f"TimeoutError: {e}")
                continue
            except Exception as e:
                logger.info(f"Failed to retrieve {url}. Error: {e}")
                continue

            chunks = []

            # Splitting text into chunks with max length of 20000 characters, but not breaking up paragraphs
            while len(text) > CHUNK_SIZE:
                chunk = text[:CHUNK_SIZE]
                last_newline = chunk.rfind("\n")
                chunks.append(chunk[:last_newline])
                text = text[last_newline:]
            chunks.append(text)

            # multiple chunks
            for chunk in chunks:
                tasks.append(
                    self.llm.do_chat_w_sys_prompt(
                        WEB_SCRAPE_PRODUCT_NAME_SYS_PROMPT_OBJ.format(
                            product_name=product, company_name=company_name
                        ),
                        WEB_SCRAPE_PRODUCT_NAME_MAIN_PROMPT_OBJ.format(
                            html=chunk,
                            product_name=product,
                            company_name=company_name,
                            main_stock=main_stock,
                        ),
                    )
                )

        first_pass = await gather_with_concurrency(tasks)
        return first_pass


async def main() -> None:
    plan_context = PlanRunContext.get_dummy()
    scraper = WebScraper(context=plan_context)
    company_name = "google"
    product_name = "mobile phone"
    result = await scraper.get_product_name(company_name, product_name, company_name)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
