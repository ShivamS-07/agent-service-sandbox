from agent_service.tools.LLM_analysis.prompts import CITATION_PROMPT, CITATION_REMINDER
from agent_service.utils.prompt_utils import Prompt

GET_PRODUCT_COMPARE_MAIN_PROMPT_STR = (
    "What are the most impactful latest {product} products from each of the following "
    "companies? {companies} The {main_stock} product should be first. Look at the stock "
    "symbol and/or ISIN to identify the stock. But remember, the company identifier must be gbi_id. "
    "Return a list where each entry is "
    "a pythonic dictionary representing the latest {product} from each of the "
    "companies, leave the company out if they are not related to the product. "
    "The dictionary should have the following entries. "
    "One entry for the company that the stock belongs to whose key is `stock_id` "
    "and value is the gbi_id of the stock, "
    "one entry for the product name whose key is `product_name`, "
    "one entry for the product release date whose key is `release_date` and "
    "the remaining entries for each of the specifications, the key being the "
    "specification and the entry being a description, missing/unavailable/undisclosed values should be n/a. "
    "For any specifications, make sure similar units are being compared "
    "against each other. Be sure to keep specifications which are "
    "relevant to {product}. Use double quotes"
    "The specification values should not be comparative "
    "to previous products. If the specification key is missing any details, "
    "fill in the details in the value field. "
    "You can place the units after the numbers in each "
    "returned result. Be sure to include any detailed specifications "
    "important to {product}. No other details, justification or formatting such as "
    "``` or python should be returned. I'm sure there are newer results, so try again. "
    "Finally, make SURE that all specification keys are the same in all dicts in the list."
    "Important! The company identifier must be gbi_id, not symbol or ISIN. "
    "Make sure that the returned dictionary is a valid python dictionary. "
)

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

GET_PRODUCT_COMPARE_MAIN_PROMPT = Prompt(
    name="GET_PRODUCT_COMPARE_MAIN_PROMPT", template=GET_PRODUCT_COMPARE_MAIN_PROMPT_STR
)

GET_PRODUCT_SUMMARY_MAIN_PROMPT = Prompt(
    name="GET_PRODUCT_SUMMARY_MAIN_PROMPT",
    template=GET_PRODUCT_SUMMARY_MAIN_PROMPT_STR + CITATION_PROMPT + CITATION_REMINDER,
)
