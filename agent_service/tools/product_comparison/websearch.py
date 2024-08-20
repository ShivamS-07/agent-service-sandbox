import io
import json
import logging
import urllib.request
from collections import defaultdict
from typing import Dict, List

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.io_type_utils import HistoryEntry, TableColumnType
from agent_service.io_types.table import Table
from agent_service.io_types.text import KPIText, TextCitation
from agent_service.tools.product_comparison.brightdata_websearch import brd_websearch
from agent_service.tools.product_comparison.constants import CHUNK_SIZE, HEADER
from agent_service.tools.product_comparison.helpers import update_dataframe
from agent_service.types import PlanRunContext
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.gpt_logging import GptJobIdType, GptJobType, create_gpt_context
from agent_service.utils.prompt_utils import Prompt
from agent_service.utils.string_utils import clean_to_json_if_needed

WEB_SCRAPE_PRODUCT_MAIN_PROMPT = """
Given part of a company's product page '{html}' from {url}.,
extract product names, and respective product features from the page.
You will be provided with a dict of product names and specs that your colleage has compiled {product_specs}.
Your task is to validate that the product information can be found on the page information you are given.
You must be critical in your evaluation of the information.
The information you need to validate will be given in dictionary format.
You should go through each key pair in the given dictionary and check if the information is present in the HTML content.
If the information is present, you should add output it, and provide justification.
If the information is present but not accurate, you should add the correct information and output and justify.
e.g. if the value is 5 but the correct value is 5 million, you should correct it.
For example, if the unit of measurement is not precisely accurate, you should correct it.
If the information is not present, you should not output it.
Then move on to the next key pair in the dictionary.
You must also include the url in your output
You should output all of this information that you have gathered
"""

WEB_SCRAPE_PRODUCT_SYS_PROMPT = """
You are a financial analyst that is doing research on company products.
You have been tasked with validating product information for a list of products.
You will be provided with part of the company's product page.
You will be provided with a list of product names and features that your colleage has compiled.
Your task is to validate the product information.
You must also include the url source in your output
"""

FINAL_VALIDATION_MAIN_PROMPT = """
You have been given a list of validation notes from your colleages {validations}.
The same information may differ between the notes.
You must go through the justification and determine the correct information.
Output the validated information as a dictionary.
If {validations} is empty, you should output an empty dictionary.
Do not output information that is not present in the notes.
But if the information is present in the notes, you should output it.
Make sure to include the url of the page that the information was found on.
Do not output any other text.
Your key should be a dict with the validated information 'value' and 'url' source, as well as the 'justifcation'.
Make sure this is the case for every key in the dict including stock_id dict
"""

FINAL_VALIDATION_SYS_PROMPT = """
You are a financial analyst that is doing research on company products.
You have been tasked with validating product information for a list of products.
You will be provided with a list of validation notes from your colleages.
These will all contain url information as well.
Your task is to validate the product information.
Do not add comments or any other text to the dict.
Make sure to use "" for strings and not ''.
Make sure that stock_id is the int gbi_id of the stock.
"""

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


class WebScraper:
    def __init__(self, context: PlanRunContext) -> None:
        gpt_context = create_gpt_context(
            GptJobType.AGENT_TOOLS, context.agent_id, GptJobIdType.AGENT_ID
        )
        self.llm = GPT(context=gpt_context, model=GPT4_O)

    async def validate_product_info(self, product_table: Table) -> Table:
        df = product_table.to_df()
        company_urls = {
            stock_id: brd_websearch(f"{product} specs", 3)
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
                (results[i].get(key, {}).get("url"), results[i].get(key, {}).get("justification"))
                for key in results[i].keys()
            ]
            urls_to_justifications = defaultdict(list)

            for url, justification in url_justifications:
                if url is not None and justification:
                    urls_to_justifications[url].append(justification)

            row_descs.append(
                HistoryEntry(
                    citations=[
                        TextCitation(
                            source_text=KPIText(val=str(url), explanation=" ".join(justifications))
                        )
                        for url, justifications in urls_to_justifications.items()
                        if url is not None
                    ],
                )
            )

        for i in range(len(df)):
            for j, col in enumerate(product_table.columns):
                if col.metadata.label != "stock_id":
                    df.iat[i, j] = (
                        results[i].get(col.metadata.label, {}).get("value", df.iloc[i, j])
                    )

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
                request = urllib.request.Request(url, headers=HEADER)
                response = urllib.request.urlopen(request, timeout=5)
                content_type = response.getheader("Content-Type")
                if "application/pdf" in content_type:
                    pdf_binary_data = response.read()
                    pdf_file = io.BytesIO(pdf_binary_data)
                    pdf_reader = PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                elif "text/html" in content_type:
                    html_content = response.read().decode("utf-8")
                    soup = BeautifulSoup(html_content, "html.parser")
                    text = soup.get_text()
                else:
                    logger.info(f"Unsupported content type: {content_type}")
                    continue
            except urllib.error.HTTPError as e:
                logger.info(f"Failed to retrieve {url}. HTTPError: {e.code} - {e.reason}")
                continue
            except urllib.error.URLError as e:
                logger.info(f"URLError: {e.reason}")
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
                            html=chunk, product_specs=product_specs, url=url
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
        return validated
