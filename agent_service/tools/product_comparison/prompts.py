from agent_service.tools.LLM_analysis.prompts import CITATION_PROMPT, CITATION_REMINDER
from agent_service.utils.prompt_utils import Prompt

GET_PRODUCT_SUMMARY_MAIN_PROMPT_STR = (
    "Look at each product within the ones listed below. {table_contents} Give a short "
    "description on each {product} product which compares it to the others as well "
    "as prior products. As well, output a bit regarding the company's trajectory "
    "with {product} products, and its status within the {product} field. "
    "You can give an overall summary afterwards. "
    "I'm only interested in the comparisons for the output, no need to restate "
    "numerical specifications or measurements from the table. "
    "No formatting symbols should be returned, only text."
    "You will also be provided with a list of citations {citations}"
)

GET_PRODUCT_COMPARE_SYS_PROMPT_STR = (
    "You are a financial analyst highly skilled at searching company products and making comparisons. You "
    "are to return the latest products with specifications from the selected "
    "companies for comparison purposes. Your result should make it easy for all "
    "users to compare the different products by making the keys consistent. It is VERY important that the products "
    "are the latest so your clients get accurate and up to date info."
)

GET_PRODUCT_COMPARE_SYS_PROMPT = Prompt(
    name="GET_PRODUCT_COMPARE_SYS_PROMPT", template=GET_PRODUCT_COMPARE_SYS_PROMPT_STR
)

GET_PRODUCT_SUMMARY_MAIN_PROMPT = Prompt(
    name="GET_PRODUCT_SUMMARY_MAIN_PROMPT",
    template=GET_PRODUCT_SUMMARY_MAIN_PROMPT_STR + CITATION_PROMPT + CITATION_REMINDER,
)

WEB_SCRAPE_IMPORTANT_SPECS_MAIN_PROMPT = (
    "Given a product type {product} and a list of companies {companies}, output up to {num_specs} of the most "
    "important specifications to compare {product} between said companies which are not price."
    "You will try to find metrics which are easily comparable for {product}"
    "You will output information in a list format where each specification is"
    'surrounded by quotes and separated by commas like ["x","y","z"].'
    "Use double quotes."
    "Do not use additional formatting such as newlines or added spaces for human readability."
    "You should only output the list, no justification."
)

WEB_SCRAPE_IMPORTANT_SPECS_SYS_PROMPT = (
    "You are a financial analyst that is doing research on company products."
    "You have been tasked with finding relevant specifications to compare similar "
    "products belonging to different companies."
    "The goal of this is to determine the pros and cons of the product from each company and output the results"
    "in a format which is easy to compare"
    "You will be given the product type as well as the companies who produce said product."
    "Your task is to output a list of exact specification comparison metrics which can be used to compare said product."
)

WEB_SCRAPE_IMPORTANT_SPECS_MAIN_PROMPT_OBJ = Prompt(
    name="WEB_SCRAPE_IMPORTANT_SPECS_MAIN_PROMPT",
    template=WEB_SCRAPE_IMPORTANT_SPECS_MAIN_PROMPT,
)

WEB_SCRAPE_IMPORTANT_SPECS_SYS_PROMPT_OBJ = Prompt(
    name="WEB_SCRAPE_IMPORTANT_SPECS_SYS_PROMPT",
    template=WEB_SCRAPE_IMPORTANT_SPECS_SYS_PROMPT,
)
