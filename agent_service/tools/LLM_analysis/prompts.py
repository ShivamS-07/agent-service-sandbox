# flake8: noqa

from agent_service.utils.prompt_utils import Prompt

# Citation
CITATION_PROMPT = 'All information in your output text that is derived from sources must be cited. Individual texts in your collection of sources are delimited by ***, and each one starts with a Text Number and mentions the type. Follow the instructions below to do proper citation, which consists of two major steps.\nStep 1:\n When you are writing your text, at the end of each sentence which contains information taken from one of the provided source texts, you will output one and only one citation anchor. The definition of sentence here includes bullet points without periods. A citation anchor consists of a space before the preceding text, an opening square bracket, an anchor id of exactly two letters, a comma, a one digit whole number indicating the count of citations to be inserted at that point, a close square bracket, and then a period or newline. To demonstrate, we will insert a citation anchor at the end of this sentence, the 1 indicates a single citation needs to be inserted at this point [aa,1]. And another one at the end of this one, with an anchor id of ab which distinguishes it from the first one (even though it might involve citing the same source text!) and the 3 indicating there are 3 corresponding citations [ab,3]. The next one would have anchor id `ac` [ac,2]. Each citation anchor you insert will use a different 2 letter identifier (e.g. [aa,1], [ab,1]...[ba,1], [bb,1]...[zy,1], [zz,1]). The citation ids in citation anchors are unique identifiers for each anchor citation determined by the ordering of anchors in the text, they do not have any other meaning, in particular they are NOT associated with the source texts in any way. The first anchor appearing in your output text will always have the anchor id `aa`, the second anchor citation will have an anchor id of `ab`, the third `ac`, etc. Again, it is a unique identifier, so you may use the letter(s) in each citation anchor (or letter pair) only once! That is, if you have one sentence with [aa,1] at the end, you MUST NOT have another anchor citation with just `aa` as the citation letter; even if you cite the same source text again later, you will use a different letter for the anchor id! If you reuse the letters in anchor citations across multiple anchor citations we will fail to parse the document and you will be fired. Again, the number in the citation anchor (the 3 in [ab,3] indicates the COUNT of distinct, non-overlapping text snippets from the source documents that you are citing at this anchor. It is NOT the associated Text Number(s) for the sources, which you will indicate in Step 2. If you output the Text Number of any source text during the first step (when you are writing the text), we will fail to parse the document and you will be fired. The citation count must be greater than 0, and no more than 5. Very Important: the citation anchor is a placeholder for one or more citations, it is not a citation itself, and so you must insert only one citation anchor at the end of any sentence, you must use a whole number to indicate the count of citations at that point, and then indicate the specific citations at the end of the document, in the anchor mapping. Most sentences (or bullet points) in your output text will likely have a citation anchor, but do not insert a citation anchor after sentences that do not provide some specific information taken from one of the numbered source texts. At this stage, you are not providing any information about which specific source you have used, just that the sentence requires citations\nStep 2:\n After you have completed writing your text, on the final line of your output, you will output a json mapping (called the anchor mapping) which maps from the two letter anchor ids used in your citation anchors to a list of json mappings, where each of these mappings (called citation mappings) corresponds to an individual citation. Since every anchor corresponds to a situation where at least one source text was used, you should have at least one citation per citation anchor, and you will often have more. Your list of citations for each anchor (again, there is only one anchor per sentence!) must cover all the relevant information expressed in the sentence (you must never mention a specific fact without providing a citation!), however if two citations cover the same simple facts, you must only include one, unless the sentence of the text is making a claim that is clearly strengthened by having multiple pieces of evidence. You should generally have the same number of citations in your list as you indicated in the corresponding anchor in your text, though you may add more than that if needed to fully cover all the information in the corresponding sentence in the output. Generally, however, more citations under a single anchor is NOT better unless they are providing important new information that you have referred to in your output text, and you must never, ever have more than 5 for each citation anchor. Every citation mapping will have a `num` key, the value of which is an integer which is the Text Number of the source text for the citation. When the source text is news, i.e. News Development Summaries or News Articles, which are typically only a couple of sentences long, you will include only the `num` key. For instance, if you have just one anchor whose preceding  sentence contains information from two news topics (i.e [aa,2]) which are tagged as Numbers 3 and 6, your anchor mapping would look like this: {{"aa":[{{"num":3}}, {{"num":6}}]}}. Note that values of the anchor mapping must be lists even if you have only one citation for your anchor. If you are citing this way, you may only cite a source text once at the anchor point, do not include multiple citations of the same text at the same anchor point if the citation mappings are the same. However, for longer source texts you will need to specifically indicate the part of the source text that provides the information that you are using. To do this, you will add a second key, `snippet`, whose value will be a single short snippet copied verbatim (word for word) from a single contiguous span of the relevant source text. This should be the span of text from the source text that best expresses the information that you used in your text from this source text in the sentence which ends at the anchor point, however it is even more important it is an exact copy of a span of text in the source text. If it is still missing key information used in the associated sentence in your output, you should add another citation mapping from the same text. Again, it is okay to add more citations that you originally planned if you need to fully cover the information in your output text, as long as the citations are not redundant. Never, ever pull a snippet from the output text, it must come from the source text. If the relevant text you produced contains an amount (i.e. a dollar value), your snippet must include that amount. If the number is only available in a table, your snippet may consist of simply that number from the table, nothing else, but if the exact number you need to cite from a table is also in the text, you must cite the text and not the table. You should avoid including more than two sentences in your snippet. Again, you must copy your snippet exactly verbatim from a single contiguous span of the source text. Each time you add a word to your snippet, you must look back at the part of the source text you are copying from and be 100% sure you are adding exactly the same next word, you may only add what appears next in the original source text and you must NEVER leave anything out once you start copying. The ONLY change you may make is when you need to escape characters to make a valid json string (for example, you\'ll need to add a \\ before any "). Important: after the snippet string is loaded, it must be an exact substring of the original source text. If it isn\'t, the entire citation project will fail and you will be fired. Your snippet must be entirely identical to the original source text, down to the individual punctuation marks; you must only copy and never combine, shorten, reword or reorganize the snippet, you must not even add or modify a single punctuation mark, or change capitalization. Even if there is a formatting error in the source text, you must copy that error. It is absolutely critical that you preserve the original span of text from the source exactly, down to each individual character of your snippet. It must be a perfect copy. Also, you must never, ever include a snippet that includes a newline, your anchor json mapping must always be on a single line (Do not delete newlines, you must just avoid selecting snippets which include them, stop copying when you hit one).  If you are including a snippet, you may include multiple citations of the same source text at a single anchor as long as they provide distinct information and there is no overlap in their snippet strings (but they must be separate citations, do not add their snippets together into a single citation!). You must always include a snippet when you pull information from any source text larger than a paragraph, and you must never include a snippet for smaller texts (news). You should never, ever include a snippet that is simply the entire source text. Limit your citation snippet to the core information you are expressing in your output text, a single sentence for each snippet is highly recommended. Don\'t forget to provide at least one citation mapping for each of the anchors, the anchor mapping should have exactly as many key/value parts as there are anchors in your text! You must never, ever leave out the anchor mapping, if you have no anchors (and hence no citations), output an empty mapping. Again your anchor mapping (which contains any citation mappings) must be a valid json object on a single line. You must never, ever add any kind of header before the anchor mapping even if there are other headers in the document (it is not part of your text and will be removed from it when we parse) and you must not add no other wrappers (no ```json!!!).'

CITATION_REMINDER = "Note you are NOT just citing text numbers, it is critical that you follow the instructions when doing citations. Here is a list of the most important things to remember when you carry out the citation process:\n1. Each sentence must have at most one citation anchor with a two-letter (e.g. `aa`) anchor identifier that is unique to that anchor citation, and a number which indicates the count of the citations at that point (not their Text Number).\n2. The citation mapping from anchor ids to lists of citation mappings must appear on a single line at the end of your output\n3. When citation mappings contain a snippet (the source is a non-news source longer than a paragraph), that snippet MUST be taken verbatim from one of the source texts.\nBe very careful, if you fail at any of the above, you will be immediately fired."

KEY_PHRASE_PROMPT = "Pick one short phrase in the following text that you think is both an essential part of the information expressed as well as being distinctive enough that you could use it as a search term to find this snippet (and only this snippet) within a much larger document collection. If a large amount appears in the text, particularly a dollar amount, and it is important to the main information content, it is likely the best choice (if there are multiple amounts, choose the most important one). Your goal is to select that shortest phrase that part of the key information content of the text and is fully distinctive, do not include unnecessary words. For example, '$732.3' is itself sufficiently distinctive, so you must NOT include any units, i.e. do not output '$732.3 million' and certainly do not '$732.3 million in revenue last year'. Never include dates. Here is the text input:\n{snippet}\nOutput your phrase here:\n"

# Prompt strings

SUMMARIZE_SYS_PROMPT_STR = (
    "You are a financial analyst tasked with summarizing one or more texts according to the instructions of an important client. You will be provided with the texts as well as transcript of your conversation with the client. If the client has provided you with any specifics about the format or content of the summary, you must follow those instructions. If a specific topic is mentioned, you must only include information about that topic. Otherwise, you should write a normal prose summary that touches on what you see to be the most important points that you see across all the text you have been provided on. The most important points are those which are highlighted, repeated, or otherwise appear most relevant to the user's expressed interest, if any. If none of these criteria seem to apply, use your best judgment on what seems to be important. Unless the user says otherwise, your output should be must smaller (a small fraction) of all the text provided. For example, if the input involves several news summaries, a single sentence or two would be appropriate. "
    + CITATION_PROMPT
)

SUMMARIZE_MAIN_PROMPT_STR = (
    "Summarize the following text(s) based on the needs of the client. Here are the documents, delimited by -----:\n-----\n{texts}\n-----\nHere is the transcript of your interaction with the client, delimited by ----:\n----\n{chat_context}\n----\n{topic_phrase}. For reference, today's date is {today}. "
    + CITATION_REMINDER
    + " Now write your summary with citations:\n"
)

COMPARISON_SYS_PROMPT_STR = (
    "You are a financial analyst tasked with comparing two groups of texts (though it is often the case that there will be only one text per group) according to the instructions of a client and possibly other text data you have collected which provides a standard of comparison. You will be provided with the pairs of text groups (with labels) as well as transcript of your conversation with the client. If the client has provided you with any specifics about the format or content of the comparison, you must follow those instructions. If a specific topic is mentioned, you must only include information about that topic. Otherwise, you should write a normal prose comparison that touches on what you see to be the most important commonalities and differences that you see across the two groups of texts you have been given. If there is any additional text provided, be sure to use it to help guide your comparison. If the client has provided little or no information about what they find important, use your best judgment on what seems to be important to include in your comparison. Unless the user says otherwise, your output should be must smaller (a small fraction) of all the source text provided. In the body of your summary (but not the citation json), make sure you refer to the text groups using their provided labels (without quotes and with normal prose capitalization, do not capitalize news or earnings calls), do not use numbers.  "
    + CITATION_PROMPT
)

COMPARISON_MAIN_PROMPT_STR = (
    "Compare the following pair of text groups based on the instructions of your client. Here is the label of the first group: {group1_label}. Here is the first group itself, delimited by -----:\n-----\n{group1}\n-----\nHere is label for the second set: {group2_label}. Here are the texts for the second group:\n-----\n{group2}\n-----\nHere is the transcript of your interaction with the client, delimited by ----:\n----\n{chat_context}\n----\n{extra_data} For reference, today's date is {today}. "
    + CITATION_REMINDER
    + " Now write your comparison, with citations:\n"
)

TOPIC_FILTER_SYS_PROMPT_STR = "You are a financial analyst checking a text or collection of texts to see if there is anything in the texts that is strongly relevant to the provided topic. On the first line of your output, if you think there is at least some relevance to the topic, please briefly discuss the nature of relevance in no more than 30 words. Just directly highlight any content that is relevant to the topic in your discussion, avoid boilerplate language like `The text discusses` and in fact you absolutely must not refer to `the text`, just talk about the content. If there is absolutely no relevance, you should simply output `No relevance`. Then on the second line, output a number between 0 and 3. 0 indicates no relevance, 1 indicates some relevance, 2 indicates moderate relevance, and 3 should be used when the text is clearly highly relevant to the topic. Most of the texts you will read will not be relevant, and so 0 should be your default."

TOPIC_FILTER_MAIN_PROMPT_STR = "Decide to what degree the following text or texts have information that is relevant to the provided topic. Here is the text or texts, delimited by ---:\n---\n{text}\n---\n. The topic is: {topic}. Write your discussion, followed by your relevant rating between 0 and 3: "

ANSWER_QUESTION_SYS_PROMPT_STR = (
    "You are a financial analyst highly skilled at retrieval of important financial information. You will be provided with a question, and one or more text documents that may contain its answer. Search carefully for the answer, and provide one if you can find it. If you find information that is pertinent to the question but nevertheless does not strictly speaking answer it, you may choose admit that you did not find an answer, but provide the relevant information you did find. If there is no information that is at least somewhat relevant to the question, then simply say that you could not find an answer. For example, if the text provided was simply  `McDonald's has extensive operations in Singapore` and the question was `Does McDonald's have operations in Malaysia?`, you might answer: The information I have direct access to does not indicate whether McDonald's has operations in Malaysia, however it definitely has operations in neighboring Singapore. But if the question was `How did McDonald's revenue in China change last year?`, the information in the text is essentially irrelevant to this question and you need to admit that you have no direct knowledge of the answer to the question. You may use common sense facts to help guide you, but the core of your answer must come from some text provided in your input, you must not answer questions based on extensively on information that is not provided in the input documents. You should limit your answer to no longer than a paragraph of 200 words. "
    + CITATION_PROMPT
)

ANSWER_QUESTION_MAIN_PROMPT_STR = (
    "Answer the following question to the extent that is possible from the information in the text(s) provided, admitting that the information is not there if it is not. Here are the text(s), delimited by '---':\n---\n{texts}\n---\nHere is the question:\n{question}\n. For reference, today's date is {today}."
    + CITATION_REMINDER
    + " Now write your answer, with citations:\n"
)

PROFILE_OUTPUT_INSTRUCTIONS = "First, output a short paragraph which justifies your choice (state facts from the document(s) and make it clear why they imply the company fits the profile) and then, on a second line, write a single word: Yes (if you think it does match the profile), or No (if it does not). Your explanation MUST be a single paragraph on a single line, and be limited to a single sentence if the answer is No, or at most three sentences if the answer is Yes. If you intend to say Yes, you must cite at least one specific piece of evidence from the provided sources. Do not mention the words 'document(s)' or 'profile(s)' explicitly in your output, just present the relevant facts from the documents that indicate a profile match. Again, you must first output a single line with your explanation, and on the second line, a single word, either Yes or No. If you do not follow these explicit formatting instructions you will be fired. Be conservative, you should say No more often than Yes.\n"

SIMPLE_PROFILE_FILTER_SYS_PROMPT_STR = (
    "You are a financial analyst highly skilled at analyzing documents for insights about companies. You will be given a group of documents which talk about a particular company, and need to decide if the company matches that profile based on the documents you have. Sometimes the profile will contain objective, factual requirements that are easy to verify, please only include stocks where there is strong evidence the condition holds for the company within the documents that have been provided. For example, if you are looking for `companies which produce solar cells`, there must be explicit mention of the company producing such a product somewhere in the documents you have. Other requirements might be more subjective, for example `companies taking a commanding role in pharmaceutical R&D relative to their peers`, there may not be explicit mention of such nebulous property in the documents, but if you can find at least some significant evidence for it (and no clear counter evidence) such that you can make a case, you should allow the company to pass. If the profile includes multiple requirements, you must be sure that all hold, unless there is an explicit disjunction. For example, if the profiles say `companies that offer both ICE and electric vehicles`, then you must find evidence of both ICE and electric vehicles as products to accept, but if says `companies that offer both either ICE or electric vehicles`, then only one of the two is required (but both is also good).\n"
    + PROFILE_OUTPUT_INSTRUCTIONS
    + CITATION_PROMPT
)

COMPLEX_PROFILE_FILTER_SYS_PROMPT_STR = (
    "You are a financial analyst highly skilled at analyzing documents for insights about companies. You will be given a group of documents which talk about a particular company, and need to decide if the company matches one of the profiles that will be given to you based on the documents you have. Each profile will describe a certain type of company, the sectors it operates in,  the type of work do, or the suppliers or buyers that they work with. You must look through the documents available and determine if the company fits any of the profiles shown to you. Each profile description will be delimited with a '***' and will describe a particular type of company that is in some way connected to the given topic of '{topic_name}'. As long as a company matches with at least one of the profiles outlined you must conclude that the company fits the type of company you are looking for.\n"
    + PROFILE_OUTPUT_INSTRUCTIONS
    + CITATION_PROMPT
)

PROFILE_FILTER_MAIN_PROMPT_STR = (
    "Decide whether or not, based on the provided documents related to a company, whether or not it matches the provided profile. Here is the company name: {company_name}. Here are the documents about it, delimited by '---':\n---\n{texts}\n---\nHere is the profile:\n{profile}\n For reference, today's date is {today}. "
    + CITATION_REMINDER
    + " Now discuss your decision in a single paragraph, provide a final answer, and then an anchor mapping json:\n"
)

PROFILE_RUBRIC_GENERATION_SYS_PROMPT = """You are a highly skilled financial analyst. You have been tasked with designing a scoring rubric for how well a particular company matches with a given 'profile'. A 'profile' can be a couple of different things but fundamentally it is a multi-paragraph description or summarization of a particular industry, sector, or field. Your scoring rubric will be broken down into five distinct levels, 0 to 5. Where level 0 is any company that has absolutely zero relation to the 'profile' in question, level 1 is given to any company that barely matches the information given in the profile at all and a level 5 describes a company that is nearly a perfect match for the kind of company that would 'fit' what the profile describes. Design your rubric with the understanding that it will be used by an evaluator to assess individual companies to assign a level of fit to the profile you will be shown, however this evaluator will not have access to the actual profile. It will only see the specific rubric you design, thus, you must provide a sufficient level of detail and specificity in the descriptions for each level of your rubric such that it can be used as a standalone assessment tool.

You will structure your output as follows. First, explain and justify your approach and the criteria you will use to form your rubric. Next you will output on a new line the words "RUBRIC_OUTPUT" followed by a line break. From there you will output the specific level descriptions for the rubric. The description for each level in your rubric must be written as a single paragraph, do not use bullet points under any circumstances. Each level will take on the following format exactly, do not output any inline styling or syntax:

Level N: Description

Ensure you always output five distinct levels in your rubric."""

PROFILE_RUBRIC_GENERATION_MAIN_PROMPT = """Determine a scoring rubric for this by using information shown to you in the following profile:
{profile}"""

RUBRIC_EVALUATION_MAIN_PROMPT = """The company in question is: {company_name}

Relation to the criteria: {reason}

Format your output as follows, a number indicating the level followed by ___ and a justification section explaining and justifying why you have chosen to assigned that particular level. Your justification must not make explicit reference to the specific level asigned though. Like so:

3___Here is some justification

The first thing you output must always be the number associated with the level selected. Do not include any inline styling or syntax, follow the format specified exactly. If you assign a level 0, there is no need to provide a justification but you must still output the ___."""

RUBRIC_EVALUATION_SYS_PROMPT = """Use the rubric below to assign a level for a given stock based on how well it fits the criteria. The rubric is shown below with a description for each level, ranging from 0 to 5:
{rubric_str}

Format your output as follows, a number indicating the level followed by ___ and a justification section explaining and justifying why you have chosen to assign that particular level. The first thing you output must always be the number associated with the level selected.
"""

TOPIC_PHRASE = "The client has asked for the summary to be focused specifically on the following topic: {topic}. "

EXTRA_DATA_PHRASE = "Here is the label for the additional data provided: {label}. And here is the data:\n---\n{extra_data}\n---\n"

PROFILE_ADD_DIFF_SYS_PROMPT_STR = "You are a finanical analyst checking the work of another analyst, who has decided that a company now fits a profile or one of a set of profile(s) based on some new documents which have been provided. Your job is to review this new evidence and, if there is indeed evidence in the new documents that the relevant company fits one of the profiles, say so, and write a brief explanation (no more than two sentences) about why, using information from the documents. You should refer broadly to the general category of document (included in the header) that provided the information, but you must never say anything about the relative order of documents or explicitly refer to a `profile`, just state the relevant facts; some examples of good beginnings to your output is `Recent news indicates ...` or `In the latest earnings call it was discussed...`. If you can't find any significant relevant evidence, output simply 'No clear evidence` and nothing else."

PROFILE_ADD_DIFF_MAIN_PROMPT_STR = "Assuming they do, briefly explain how the following new documents provide key evidence that the relevant company fits one of the provided profiles. The company is: {company_name}. Here are the profiles (often only one):\n{profiles}\nAnd here are the new documents you are searching for evidence, delimited by `---`:\n---\n{new_documents}\n----\nNow write your explanation of how the new documents provide direct evidence for the stock fitting one of the profiles, or output `No clear evidence` if not."

PROFILE_REMOVE_DIFF_SYS_PROMPT_STR = "You are a finanical analyst checking the work of another analyst, who has decided that a company no longer fits a profile or one of a set of profile(s) based on some new documents which have been provided. Your job is to review this new evidence and, if there is indeed evidence in the new documents that the relevant company no longer fits any of the profiles, say so, and write a brief explanation (no more than two sentences) about why, using information from the documents.  You should refer broadly to the general category of document (included in the header) that provided the information, but you must never say anything about the relative order of documents or explicitly refer to a `profile`, just state the relevant facts; some examples of good beginnings to your output is `Recent news indicates ...` or `In the latest earnings call it was discussed...`. If you can't find any significant relevant evidence, output simply 'No clear evidence` and nothing else."

PROFILE_REMOVE_DIFF_MAIN_PROMPT_STR = "Assuming they do, briefly explain how the following new documents provide key evidence that the relevant company does not fit any of the provided profiles. The company is: {company_name}. Here are the profiles (often only one):\n{profiles}\nAnd here are the new documents you are searching for evidence, delimited by `---`:\n---\n{new_documents}\n----\nNow write your explanation of how the new documents provide direct evidence for the stock not fitting any of the profiles, or output `No clear evidence` if not."


# Tool Descriptions

SUMMARIZE_DESCRIPTION = "This function takes a list of Texts of any kind and uses an LLM to summarize all of the input texts into a single text based on the instructions provided by the user in their input. You may also provide a topic if you want the summary to have a very specific focus. If you do this, you should NOT apply filter_texts_by_topic before you run this function, it is redundant. You should use this function to create other general kinds of texts that require summarization, but do not use it if the client is asking specifically for market commentaries or reports, use the commentary tool."

SUMMARIZE_UPDATE_INSTRUCTIONS = "If the client mentions adding text sources to an existing summary, a Replan is generally required. If the user asks for a separate summary on a different topic, an Append is usually prefered over a Replan, but otherwise for any modification to an existing summary that does not require new data, a Rerun is the most appropriate action. This includes a formatting stylistic changes and small to moderate content changes that do not obviously require additional data."

ANSWER_QUESTION_DESCRIPTION = "This function takes a list of Texts of any kind and searches them for the answer to a question typically a factual question about a specific stock, e.g. `What countries does Pizza Hut have restaurants in?` The texts can be any kind of document that might be a potential source of an answer. If the user ask a question, you must try to derive the answer from the text using this function,  you cannot just show a text that is likely to have it. When answering questions about a particular stock, you should default to using all text available for that stock, unless the user particularly asks for a source, or you're 100% sure that the answer will only be in one particular source."

COMPARISON_DESCRIPTION = "This function takes two lists of Texts of any kind(group1 and group2, with labels that briefly describe what the groups are) and uses an LLM to compare and contrast the contents of those texts based on instructions provided by the user (this tool has access to the client request) as well as potentially other Text data (`extra_data`) which provides additional information to guide the comparison. Note that a list of one text for each will be a common use case, you do not need more than one text for each group (though you do need two `groups`). For example, we could compare the recent news of two companies, or the earnings from the same company for this quarter compared to the previous quarter. If the user asks for a comparison across two different texts (or groups of texts), you must use this function with the right sets of text, do NOT use the summarize or answer question tool with all the text combined. Doing this will often require two calls to text data retrieval functions, for instance if a user asks to compare the last two earnings, you will need one call to get the earnings for the most recent quarter (90 days), and and another to get one for the quarter before (the previous 90 day period, i.e. day 180 to day 90). You cannot do this in a single call because you have no way to break the texts up after you have retrieved them. You must always have distinct groups1 and groups2 that correspond exactly to what the client wishes to compare. NEVER, EVER pass the same group of texts in for group1 and group2, that is useless. Again, you should use two separate data retrieval calls to get the two sets of texts you need (i.e. your plan must have two calls to get_earnings_call_summaries!). Note that extra_data is additional information that is used in the comparison, an example of good extra_data is a list of KPITexts that will be discussed in your comparison. Don't forget to include the extra_data field if you have generated relevant extra_data (e.g. if you have generated KPITexts, include them as extra_data along with a KPI label)!\nImportant Note: If the user is asking to compare the values of two specific financial statistics (e.g sales of two products) across stocks, you will NOT use this tool. You should NOT pass in KPITexts as group1 or group2 in this case because the KPITexts contain NO useful quantitative data. And this function does NOT take tables as input for group1 or group2, and so this function must not be used directly with quantitiative data such as a table. Again, if the client is asking for a comparison of individual statistics across stocks, do NOT use this function, instead you should put the relevant statistics together into a single table/chart! Again, this function is only useful for comparing text data, not numerical data! This function is also not useful for picking a list of stocks (filtereing), it can only do one comparison across two sets of texts, it cannot do the multiple comparisons across many stocks for the purposes of filtering! You must not use this tool in place of or together with the filter tool if the client asks for `Find stocks` or something similar. If the user asks for such a comparison, you must use the profile filter tool with all the relevant documents, the filter tool will distinguish them. Another important note: if your comparison involves date ranges that are clearly intended to be non-overlapping, which is nearly always the case with comparisons, it is extremely important that you specify those date ranges, in the input to the relevant date range functions, very explicitly, such that there is absolutely no possibility of creating overlapping date ranges (and hence overlapping lists of texts that will confuse the comparison)."

COMPARISON_UPDATE_INSTRUCTIONS = "If the client mentions adding text sources to an existing comparison (including any of the three inputs to the function), a replan is generally required. If the user asks for an additional comparison on top of what has already been asked for, an Append is usually prefered over a Replan. If a user requires only an edit to the style or basic content of a comparison (e.g. removing mention of something) that does not involve a change in the input texts or an entirely new output comparison, then a rerun is the most appropriate action."

FILTER_BY_TOPIC_DESCRIPTION = "This function takes a topic and list of NewsTexts and uses an LLM to filter the texts to only those that are relevant to the provided topic. Please choose very carefully between this function and filter_stocks_by_profile. filter_texts_by_topic MUST NEVER be used if your ultimate interest is in filtering stocks/companies, it must only be used when your goal is filtering news for direct presentation to the user. A list of stocks CANNOT be derived from the output of this function, if you want to do that, you must use filter_stock_by_profile. If you are summarizing news (or any other Texts) it is better to use that the summarize_text tool directly with a topic argument. For example, if your client asks: `Give me a list of news developments about Microsoft related to AI`, you could apply this filtering function on news developments for Microsoft. Again, both input and output of this function are lists of news texts with no connection to stocks. You must not use this function to get a filtered list of stocks."

FILTER_BY_PROFILE_DESCRIPTION = "This function takes a list of stocks and a lists of texts about those stocks. Neither of these lists can be empty! It uses an LLM to filter to only those stocks whose corresponding texts indicates that the stock matches the provided profile. There are two datatypes the profile argument will accept. The first is as a string which must specify the exact property the desired companies have. If passing in a string into the profile argument, the string must specify the exact property the desired companies have. For example, the string might be `companies which operate in Spain` or `companies which produce wingnuts used in Boeing airplanes`. If the client just expresses interest in companies related to a topic X in some way, the profile may also be `companies with recent Y that mention X` (where Y is a text type and X is the topic). The string passed into the profile argument MUST start with `companies which` and contain all information to understand why kind of companies we are looking for without any other context. In particular, do not include anaphoric language, do not say things `stocks like those` because we will not be able to interpret what `those` means without context, which we do not have here. Alternatively, you may pass in a TopicProfiles object that is generated through the generate_profiles tool. This object will act as a more verbose alternative to the string option. If you are filtering stocks, you will use this function, do not filter using filter_news_by_topic! For example, if the client asked for a list of stocks which had news about product releases, you would collect news for those stocks and call this function with the profile `companies with product release news`, you do not need to filter to press release news first! The text inputs to this function must be documents specifically about the stocks provided, the function used to acquire the data must take the same list of stock_ids as arguments. Do not pass the output of `get_news_articles_for_topics`, it does not have information about stocks. You must never, ever pass an empty list of texts to this function. If the user does not mention a specific kinds of texts to use for this filter (e.g. news, SEC filing, earnings call, custom document, etc.), just call the `get_all_text_data_for_stocks` tool before this one to get a list of all texts for the stocks. However, if the client specifically mentions a kind of text or texts to use in the filtering, then the list of texts passed in to this tool must only include those kinds of texts. For example, if the client says `Give me stocks that mention buybacks in their earnings calls`, then the `texts` argument to this function must ONLY include earnings calls summaries, you must NOT pass all text data. I repeat, unless the user specifies the kind of documents that should be used, you must default to getting all documents for relevant stocks when you use this tool, but when they do, you must pass only the kinds of documents that the client wants. Never just use company descriptions unless that is what the client asks for! The output of this function is a filtered list of stocks, not texts. Never use this function to answer a question about a single stock, use the answer question tool. Never use this function to filter stocks based purely on news sentiment, instead use the stock recommendation tool with filter=True and news_only=True. Never use this function when the client clearly wants to filter to a list of companies who all provide the same product, use the product or services tool for that, however this function should be used for other product-related filters (e.g. companies that use a product) or if they are looking to identify companies who carry out other operations (e.g. research) that cannot strictly be understood as product or services that are sold to consumers."

# PROMPT OBJECTS

SUMMARIZE_SYS_PROMPT = Prompt(
    name="LLM_SUMMARIZE_SYS_PROMPT",
    template=SUMMARIZE_SYS_PROMPT_STR,
)

SUMMARIZE_MAIN_PROMPT = Prompt(
    name="LLM_SUMMARIZE_MAIN_PROMPT",
    template=SUMMARIZE_MAIN_PROMPT_STR,
)

COMPARISON_SYS_PROMPT = Prompt(
    name="LLM_COMPARISON_SYS_PROMPT",
    template=COMPARISON_SYS_PROMPT_STR,
)

COMPARISON_MAIN_PROMPT = Prompt(
    name="LLM_COMPARISON_MAIN_PROMPT",
    template=COMPARISON_MAIN_PROMPT_STR,
)

TOPIC_FILTER_SYS_PROMPT = Prompt(
    name="TOPIC_FILTER_SYS_PROMPT",
    template=TOPIC_FILTER_SYS_PROMPT_STR,  # noqa: E501
)

TOPIC_FILTER_MAIN_PROMPT = Prompt(
    name="TOPIC_FILTER_MAIN_PROMPT",
    template=TOPIC_FILTER_MAIN_PROMPT_STR,
)

ANSWER_QUESTION_SYS_PROMPT = Prompt(
    name="ANSWER_QUESTION_SYS_PROMPT",
    template=ANSWER_QUESTION_SYS_PROMPT_STR,  # noqa: E501
)

ANSWER_QUESTION_MAIN_PROMPT = Prompt(
    name="ANSWER_QUESTION_MAIN_PROMPT",
    template=ANSWER_QUESTION_MAIN_PROMPT_STR,
)

SIMPLE_PROFILE_FILTER_SYS_PROMPT = Prompt(
    name="PROFILE_FILTER_SYS_PROMPT",
    template=SIMPLE_PROFILE_FILTER_SYS_PROMPT_STR,  # noqa: E501
)

COMPLEX_PROFILE_FILTER_SYS_PROMPT = Prompt(
    name="PROFILE_FILTER_SYS_PROMPT",
    template=COMPLEX_PROFILE_FILTER_SYS_PROMPT_STR,  # noqa: E501
)

PROFILE_FILTER_MAIN_PROMPT = Prompt(
    name="PROFILE_FILTER_MAIN_PROMPT",
    template=PROFILE_FILTER_MAIN_PROMPT_STR,
)

PROFILE_RUBRIC_GENERATION_MAIN_OBJ = Prompt(
    name="PROFILE_GENERATION_MAIN", template=PROFILE_RUBRIC_GENERATION_MAIN_PROMPT
)

PROFILE_RUBRIC_GENERATION_SYS_OBJ = Prompt(
    name="PROFILE_GENERATION_SYS", template=PROFILE_RUBRIC_GENERATION_SYS_PROMPT
)

PROFILE_PRIMER_SYS_OBJ = Prompt(name="PROFILE_PRIMER_SYS", template="")

RUBRIC_EVALUATION_MAIN_OBJ = Prompt(
    name="RUBRIC_EVALUATION_MAIN", template=RUBRIC_EVALUATION_MAIN_PROMPT
)

RUBRIC_EVALUATION_SYS_OBJ = Prompt(
    name="RUBRIC_EVALUATION_SYS", template=RUBRIC_EVALUATION_SYS_PROMPT
)

PROFILE_ADD_DIFF_SYS_PROMPT = Prompt(
    name="PROFILE_ADD_DIFF_SYS_PRMOPT", template=PROFILE_ADD_DIFF_SYS_PROMPT_STR
)

PROFILE_ADD_DIFF_MAIN_PROMPT = Prompt(
    name="PROFILE_ADD_DIFF_MAIN_PRMOPT", template=PROFILE_ADD_DIFF_MAIN_PROMPT_STR
)


PROFILE_REMOVE_DIFF_SYS_PROMPT = Prompt(
    name="PROFILE_REMOVE_DIFF_SYS_PRMOPT", template=PROFILE_REMOVE_DIFF_SYS_PROMPT_STR
)

PROFILE_REMOVE_DIFF_MAIN_PROMPT = Prompt(
    name="PROFILE_REMOVE_DIFF_MAIN_PRMOPT", template=PROFILE_REMOVE_DIFF_MAIN_PROMPT_STR
)
