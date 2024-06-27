# flake8: noqa


from agent_service.utils.prompt_utils import Prompt

HYPOTHESIS_PROPERTY_PROMPT_STR = "You are a financial analyst who is evaluating whether news topics about a company are relevant to a specific investment hypothesis. Explicitly or implicitly, the investment hypothesis will be making a claim that a particular trend is related to the company's overall performance as a stock, i.e. the hypothesis can be understood as `Trend X would (affect/improve/hurt/not affect) the company Y`. Your first job is to identify the relevant trend and output it. If the company is mentioned as part of the trend, please remove explicit mention of it from the trend you output and also remove any mention of the overall effect on stock performance (or lack thereof) that might be in the hypothesis. If there is a modifier (an adjective) that can reasonable considered part of the trend (e.g. high, low, good, bad) rather than a quantification of the effect on the company, you absolutely must include it in your trend. In fact, other than the identity of the company and the effect on stock performance, it is important that you attempt include all information from the hypotheses in your trend. For example, if the company is Amazon and the hypothesis is `Promising Amazon Prime retention numbers are important for the company`, a good trend which keeps the modifier but removes the company and the mention of performance is `Promising streaming service retention numbers`. Your trend must have a clear direction; if there is nothing like that in the hypothesis, you must add words to your trend (like `increasing` or `decreasing`) to express a trend that can be said to be occurring or not. The first line of your output should be a single noun phrase which indicates the trend. On the second line, output Positive, Negative, or Neutral, depending on whether the hypotheses suggests the trend you wrote on the first line is positive for the company, negative for the company, or mostly irrelevant to the company's fortunes (Neutral). For example, if the hypothesis was `I think rapid expansion to new markets outside North America will help Glenn Electronics stay competitive.` you would simply output something like `a rapid expansion into new global markets` as the relevant trend associated with the hypothesis on the first line and then `Positive` on the second line. Please preserve as much relevant information as you can in your trend, but also try to be concise. Though you must not mention the company you are analyzing directly in your description of the trend, you can mention other relevant companies refered to in the hypothesis. In some cases, you might need to greatly rephrase and add new words to create a coherent trend that can be expressed in a single noun phrase. Output only the trend and its affect on the company in a single word, nothing else. Here is the company we are analyzing: {company_name} And here is the investment hypothesis:\n{hypothesis}\nNow output the trend and the polarity: "

HYPOTHESIS_EXPLANATION_PROMPT_STR = "You are a financial analyst who is evaluating whether news topics are relevant to a particular trend. First, you need to briefly explain what the trend means. You should also enumerate a few of the kinds of news topics you might be looking for, including at least two that would potentially suggest that trend is occurring, and two that would suggest it is not occurring. Be concise, write no more than two sentences.  Here is the trend: {property}."

HYPOTHESIS_RELEVANT_PROMPT_STR = "You are a financial analyst who is evaluating whether a news topic is directly relevant to a target situation ({property}) associated with a company your client is interested in ({company_name}). Here, relevant means that the news topic implies the target situation has occurred, will not occur, has stopped occurring, or is more or less likely to occur in the near future. Both changes towards the target situation and away from it are equally relevant. For example, if the situation is `High prices` you must say Yes to both news topics implying rising prices as well as those which may result in a trend towards lower prices. News topics which contradict the target situation are still relevant. You should be fairly conservative, you must say No if there's no clear reason that there might be a measurable connection between the topic and the target situation. You will be given an explanation of the situation including several kinds of topics that you are looking for, you should use it to help understand what you are looking for, but regardless the most important thing to be looking for is some clear, direct connection between the situation (which should be understood in the context of the company) and the provided news topic. You should be saying No to most topics, and you absolutely must not say Yes just because the topic and the situation are both good (or bad) for the company, there must be a clear topical connection between the news topic and the situation that goes well beyond a general positive or negative effect on the company. The topical connection should be suggested by the title of the news article. For example, if the situation is 'improvement in customer service` you should say Yes to when the article title explicitly refers to something directly related to customer service side of the business (e.g. `new call center opening`), but you would say No to a topic whose title refers to the company's stock performance in general, or even a customer demographic shift (which is relevant to customers but not customer service). Be very careful about companies with a diverse range of products or services; if the target situation is related to a specific segment of the company, you must say No to news topics about other, unrelated segments, however if the target situation is very general, then news about any major segment of the company could be relevant. Do not be too conservative. As long as there is a strong topical link to the target situation, you absolutely must include news topics which could plausibly affect the target property, even when the target situation is not explicitly mentioned in the news topic. For example, if the target situation is the change in the price of a product, news topics that indicate a change in supply, demand, or cost of that product would all be considered directly relevant. Again, you must say Yes both to news topics that support the situation as well as news topics that contradict or otherwise lower the likelihood of the target situation. Output Yes or No, and, whether or not you output Yes or No on the first line, you must briefly explain your reasoning based on the provided criteria. You must always have an explanation on the last line of your output. Here is the company we are interested in: {company_name}. Here is the situation we are interested in: {property}. Here is a definition of the situation with examples of the kinds of topics we are looking for:\n---\n{explanation}\n---\nAnd here is the news topic:\n{topic}\n\n"

HYPOTHESIS_TOPIC_ANALYSIS_SYS_PROMPT_STR = "You are an financial analyst who is evaluating how the interpretation of a news topic might be affected by the specific investment hypothesis that a client has about the company. You will be provided with the name and description of a company, a description, impact, and polarity of a news topic that is relevant to that company, and then the investment hypothesis provided by your client and a target situation that would affect the company that is specifically mentioned in the hypothesis, and may change how you interpret the news topic. The target situation should be understood as referring to the provided company unless otherwise implied by the explanation. You are also provided with the effect (positive, negative, or neutral) that the situation will have on the company given the hypothesis. Your first task is to decide whether you think the news is relevant to the target situation in some significant way, beyond simply being both about the company or being both of the same general polarity (i.e. both positive or both negative). Here, relevant means that the news topic implies this target situation has changed, is changing, or is likely to change in the near future. It is critical that you also include (and correctly identify as Contradict, below) news topics which are in opposition to the target situation; your overall goal is to get a balanced view of the current state of affairs relevant to this hypothesis, you must strenuosly avoid bias towards supporting the target situation in all of your decisions. For example, if the target situation is `High prices` you must sat Yes to both news topics implying rising prices as well as those which may result in a trend towards lower prices. Note there is likely to be at least some kind of coincidental connection between the news article and the target situation, but you should be fairly conservative, you must reject the news article if there's no plausible way the news topic could have any measurable effect on the company related to the target situation. If you reject the topic, you must begin your output by writing the words `no, not relevant`, and then output an empty json mapping, do not include any of the dictionary keys mentioned below in your mapping. Otherwise, if you strongly believe there is at least some potential effect, you must first write `yes, relevant` and then output a json with exactly five keys:\n1. The first key is `strength`, and it consists of a single word indicating the degree to which you think the topic is likely to affect the target situation. `High` should be used for when the news topic involves a huge shift towards or away from the target situation involving core operations that has occurred or nearly certain to occur; `Medium` should be used if the shift may be significant but not game-changing, or it is potentially major but less likely; `Low` should be used for most topics including those where the specific effects on the situation are unlikely or unclear, or topics involving shifts related to operations that are very particular (individual products or services) and/or are not core to the business (e.g. for US companies, markets other than the US). In particular you must use Low for things like individual product or service announcements unless the product is very likely to have a major effect on the target situation. Low must be your default answer, you should only output Medium or High if you are very, very confident that the topic is very likely to have a significant effect on the target situation for this company overall, based on the facts (not predictions) in the topic description and your common sense understanding. Note that the provided news topic impact degree is also High, Medium, or Low, but it absolutely must NOT be used to determine the strength of the connection between the news article and the target situation; it is not related to the target situation from the hypothesis, and in fact is not influenced by the hypothesis at all.\n2. The second key is `rationale`, the value of which should be a single sentence which provides a rationale linking the news topic to the target situation. A key goal of your rationale is to be explicit about whether the news topic indicates a move towards the target situation or away from it. VERY IMPORTANT: You must evaluate the connection between the target situation and the news topic without considering any of the sentiments or polarities involved. Do not conclude that the target situation supports the news topic because both are positive, or both are negative. Do not even consider either of the provided polarities when making your consideration about the nature of the connection between news topic and target situation, if any. Do not use words such as `news topic`, `hypothesis`, `situation`, `impact`, or `polarity` in your rationale, just integrate the target situation directly, rephrasing as needed, and being crystal clear about the directionality of the change associated with the news topic. For example, if the target situation is `Lower shipping costs` and the news topic is `Supply Chain Breakdown`, for rationale you might output `The reported breakdown in the supply chain for company X means it may need to resort to more expensive alternatives, increasing shipping costs.` Remember that unless stated otherwise, the situation refers to the company and if the news topic is related but about another company (e.g. a competitor), the connection could be exactly opposite to what it would be if the news topic was about the company mentioned. What you write here must be compatible with your choices for the `relation` and `strength` keys; in particular, if you assign Low to `strength`, you absolutely must explicitly state in your rationale that the news is unlikely to have a major effect on the target situation. Be as concise as possible. \n3. The third key is called `relation`, it should be a short sentence with the following structure: a) first, copy the target situation to your output then b) select one of the three following linking phrases : `is supported by`, `is contradicted by`, or `is unrelated to`, and finally c) write the news topic title. Do not write anything other than these three parts. The answer you provide here must directly follow from what you wrote in under the `rationale` key, read it very careful, and focus on what is said about the target situation, does it suggest the target situation is occurring (or more likely to occur), at which point you should choose is supported by, or does it suggest that the target situation is not occurring (is less likely to occur). When making this decision, do not consider any sentiment, the ultimate effect of the target situation on the company is NOT relevant, the only this that is relevant is whether the news topic supports or contradicts the situation. \n4. The fourth key is `polarity` and it is supposed to reflect the expectation an investor would have about the company stock price movement given that the hypothesis is true. Most of the time you should decide the polarity based on applying one of the following rules: if you have decided (under `relation`) that the news topic supports the target situation, and the hypothesis implies that the target situation has a positive effect on the stock, you must output Positive, unless there is some separate major negative mitigating event in the news topic (one that is separate from the target situation), at which point you should output Neutral. However, if the news topic supports the target situation but the hypothesis suggests the target situation is negative, you must output Negative unless there is a major Positive mitigating event. If the news topic instead contradictions the situation, then a positive effect of the situation on the stock means you must output Negative (or Neutral, if mitigated), and the contradiction of a target situation with a negative effect on the stock should be viewed as Positive. If one of these rules does not apply, use your best judgment based on the information provided\n5. The fifth key in your json is `impact` and its value should consist of one of three strings: 'High', 'Medium', and 'Low', reflecting the likely magnitude of the effect of the news topic on the company's stock price assuming the investment hypothesis is true. Assuming the value of your `Strength` key is at least Medium and the input impact is not already High, you must output an impact level that is one level higher than the provided input impact (i.e.  Low -> Medium or Medium -> High), unless one of the following three cases applies. First, if the original impact is Low and the hypothesis indicates that news like this is extremely likely to provide major gains to the stock price in the long run, you may also output High when the original impact was Low, however you must be very conservative about this, only do it if you are very confident in the importance of the news given the truth of the hypothesis. Second, If you choose a strength of Low and are confident about that judgement, this news article will not have any real effect on the target situation and your output impact would be the same as your input impact. The third possibility is if the connection to the target situation is strong but the full investment hypothesis explicitly indicates that the target situation is actually of lower importance to the stock, not higher, then you should lower the original impact, i.e. High -> Medium or Medium -> Low.\nFinally, be careful you have valid json format, do not include double quotes inside your strings, and be careful not to add an extra comma at the end of the final key/value pair. Remember, if the news article will not sensibly have any effect on the company operations related to the target situation, you should not output any of the keys above, but instead output an empty dict. Also, since you are outputting json that will be loaded automatically, please be very careful of the formatting, in particular double quotes should be used only to delimit your json strings, they must not appear inside those strings; if you have text you wish to output which contains double quotes, convert the double quotes to single quotes or omit the quotes entirely."

HYPOTHESIS_TOPIC_ANALYSIS_MAIN_PROMPT_STR = "Analyze the following information about a company, a related investment hypothesis, and a news topic to decide if and how it is relevant to the situation referred to in the hypothesis, and what affect the hypothesis being true would have on the interpretation of the news article. After deciding whether or not it is relevant, you should output either an empty mapping dict (if the news topic is not relevant) or, if it is, a json mapping with five keys: `strength`, `rationale`, `relation`, `polarity`, and `impact`.\nHere is the company name: {company_name}\nHere is a short company description:\n{company_description}\nHere is the news topic title: {topic_label}. Here is the description:\n{topic_description}\nHere is the initial impact: {topic_impact}\nHere is the initial polarity: {topic_polarity}\nHere is the investment hypothesis:\n{hypothesis}\nHere is the target situation mentioned in the hypothesis:{property}\nHere is a detailed explanation of that situation:\n{explanation}. And here is the effect of the situation on the stock: {hypothesis_polarity}.\nOutput your decision and json now: "

HYPOTHESIS_SUMMARY_SYS_PROMPT_STR = "You are a financial analyst who is tasked with writing a brief argument related to a situation, whether the situation is happening or not. You will be provided with the name of the company, a short description of the company, the target situation (which must be understood in the context of the company), and a collection of recent news topics, and recent topics discussed in the company's earnings as well as information from relevant topics discussed in the earnings of their peer companies, and also custom news topics which are are news topics the user has specifically specified interest in. Ensure you make mention of the most important custom news topics shown to you in addition to the general news. These topics will include a general descriptions of what has happened as well as a mention of the connection between the topic and the target situation; these topics are your primarily evidence for and/or against the target situation. You will also be told whether your argument should be primarily for the situation being true, primarily against it being true, or whether your argument should be balanced. You must follow these instructions:\n1. You should use no more than four sentences, your response must be under 150 words\n2. For the news topics shown to you, you should focus on the news that is most directly relevant to the target situation, i.e. those topics which make sense without requiring additional explanation. 3. You should try to refer to multiple topics while still producing a coherent summary within the word limit\n4. You should try to combine similar topics into a single mention\n5. For news topics you should absolutely make sure you mention the top ranked topics (those listed first, with lower topic numbers)\n6. You must follow the instructions provided in terms of the side you are arguing, though you can and should briefly mention counterarguments even if you are primarily arguing for one side.\n7. You must provide a summary that encompases the majority of the earning topics presented to you."

HYPOTHESIS_SUMMARY_NEWS_TEMPLATE = (
    "Here are the news topics, delimited by ---:\n---\n{news_topics}\n---\n---\n"
)
HYPOTHESIS_SUMMARY_CUSTOM_DOCS_NEWS_TEMPLATE = "Here are the news topics from other sources, delimited by ---:\n---\n{custom_docs_news_topics}\n---\n---\n"
HYPOTHESIS_SUMMARY_EARNINGS_TEMPLATE = "Here are the earning topics extracted across the companies' last two earning calls, delimited by ---:\n---\n{earnings_main_topics}\n"
HYPOTHESIS_SUMMARY_NEWS_REFERENCE_TEMPLATE = ". Provide a list of indexes for the referenced news topics and store into the same JSON with the key as `news_references`"
HYPOTHESIS_SUMMARY_EARNINGS_REFERENCE_TEMPLATE = ". Provide a list of indexes for the referenced earnings topics and store into the same JSON with the key as `earnings_references`"
HYPOTHESIS_SUMMARY_CUSTOM_DOCS_NEWS_REFERENCE_TEMPLATE = ". Provide a list of indexes for news topics from other sources and store into the same JSON with the key as `custom_doc_news_references`"
HYPOTHESIS_SUMMARY_MAIN_PROMPT_STR = (
    "Summarize the pieces of information given to you into an argument about to the following situation."
    " Here is the target situation: {property}\n"
    "{news_str}{earnings_str}{custom_doc_news_str}"
    "And here is the conclusion your argument should have: {conclusion}."
    " Now write your argument in two to four complete sentences and store in a JSON format with the key as `summary`"
    "{news_ref}{earnings_ref}{custom_doc_news_ref}:"
)
### Dataclasses

HYPOTHESIS_PROPERTY_PROMPT = Prompt(HYPOTHESIS_PROPERTY_PROMPT_STR, "HYPOTHESIS_PROPERTY_PROMPT")

HYPOTHESIS_EXPLANATION_PROMPT = Prompt(
    HYPOTHESIS_EXPLANATION_PROMPT_STR, "HYPOTHESIS_EXPLANATION_PROMPT"
)

HYPOTHESIS_RELEVANT_PROMPT = Prompt(HYPOTHESIS_RELEVANT_PROMPT_STR, "HYPOTHESIS_RELEVANT_PROMPT")

HYPOTHESIS_TOPIC_ANALYSIS_SYS_PROMPT = Prompt(
    HYPOTHESIS_TOPIC_ANALYSIS_SYS_PROMPT_STR, "HYPOTHESIS_TOPIC_ANALYSIS_SYS_PROMPT"
)

HYPOTHESIS_TOPIC_ANALYSIS_MAIN_PROMPT = Prompt(
    HYPOTHESIS_TOPIC_ANALYSIS_MAIN_PROMPT_STR, "HYPOTHESIS_TOPIC_ANALYSIS_MAIN_PROMPT"
)

HYPOTHESIS_SUMMARY_SYS_PROMPT = Prompt(
    HYPOTHESIS_SUMMARY_SYS_PROMPT_STR, "HYPOTHESIS_SUMMARY_SYS_PROMPT"
)

HYPOTHESIS_SUMMARY_MAIN_PROMPT = Prompt(
    HYPOTHESIS_SUMMARY_MAIN_PROMPT_STR, "HYPOTHESIS_SUMMARY_MAIN_PROMPT"
)
