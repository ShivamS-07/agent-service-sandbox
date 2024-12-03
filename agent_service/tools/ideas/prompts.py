# flake8: noqa

from agent_service.tools.LLM_analysis.prompts import CITATION_PROMPT, CITATION_REMINDER
from agent_service.utils.prompt_utils import Prompt

INITIAL_BRAINSTORM_PROMPT_STR = "You are a financial analyst reviewing a large body of text data to try to identify {ideas} that will satisfy the particular needs of one of your clients. You will be provided with a specific definition of the kind of {ideas} you are looking for. In this initial round, your goal is to brainstorm as many distinct {ideas} you can come up with that satisfy the client's request and have some grounding in the text data provided. To be clear, everything you come up with must be based on information from at least one of the provided texts (which are often larger documents broken down into sections). The more relevant documents for each {idea}, the better, ideally all your {ideas} will come from multiple documents. Do not brainstorm {ideas} that are not based on facts in your text data, if you do this you will be fired. High diversity across the relevant {ideas} is extremely important, but high diversity of sources for each {idea} is also extremely desirable. If there is diversity in your text data in terms of relevant stocks, relevant text types, relevant dates, or any other clear properties, you should first list {ideas} those which appear in as many different types as possible. Each of your ideas must have at least one unique source, if you run out of sources, you should stop listing ideas. If a company is specifically mentioned in the {idea} definition below, you may mention it in your {ideas}, but otherwise you must avoid any reference to any specific company in any of {ideas}. Seriously, DO NOT mention the name of a company other than one mentioned in the idea definition or you will be fired!!! Your output will consist of a json list of your {ideas}, where each item of the list will be a dictionary with two keys. The first key is `{idea}` and will consist of a short phrase of 3 to 5 words which uniquely identifies the {idea}, followed by a colon and then a brief explanation of no more than 30 words. You must not go into any significant detail, this will be done in a later step. The second key is `sources`, a list of integers corresponding to the Text Numbers of the source documents which mention that {idea}. `sources` must always be a list even if you have one source, and it must never be empty, every one of your {ideas} must be based on some relevant reference in a source text. You will be fired if you brainstorm {ideas} with no source documents, or if you list source document Text Numbers where the relevant source documents do NOT actually mention the corresponding {idea}. Do not include more than 10 sources. As long as each {idea} is reasonable, distinct, and supported by evidence, you must include as many as you can, though it is better to include just a few excellent, well-supported {ideas} than to produce {ideas} with little or no grounding in the text data provided. Be very careful not to duplicate {ideas}, each of your short initial phrases must be entirely distinct! You should use whatever wording the client uses to describe what they are asking for. Here is the definition of {ideas} you are looking for: {idea_definition}.\n---\nHere are the texts, also delimited by ---:\n---\n{texts}\n---\nNow you must brainstorm as many {ideas} as you can from the texts that satisfy the client's requirements, and include the numbers of your source document you got your {ideas} from. Output your {ideas} json now:\n"

INITIAL_BRAINSTORM_PROMPT = Prompt(INITIAL_BRAINSTORM_PROMPT_STR, "INITIAL_BRAINSTORM_PROMPT")

IDEA_CLUSTER_SYS_PROMPT_STR = "You will be given a numbered list of pairs of brainstormed ideas related to a particular finance need. The overall interest of this user to come up with a nonredundant list of {items}. Your goal is to identify ideas that are similar enough that they can be collapsed into a single idea. Since all the ideas are aimed at satisfying the above user need, that connection is NOT a commonality to say Yes, there should be some other major semantic connection. The ultimate goal is to avoid redundancy, so please be do say Yes to in any case where you think the main idea largely overlaps and can be easily combined into a single idea, even if the specific details of the two are a bit different. You should give your answer for each pair by writing the number for the pair, a space, and then either Yes or No, on one line, e.g.\n10 No\n or \n10 Yes\n. Do not include a period between the number and your Yes/No. Do not include any other information, or any explanation."

IDEA_CLUSTER_MAIN_PROMPT_STR = "Decide whether or not the following pairs of ideas are similar enough to collapse into a single idea. You must output lines consisting of the number for one of the provided pairs and either Yes or No, separated by a space. Here are the pairs of ideas:\n\n{text_pairs}"

IDEA_CLASSIFY_SYS_PROMPT_STR = "You will be given a list of one or more related brainstormed ideas, and then another, separate idea. The overall interest of this user to collapse ideas into broadly related groups, related to a particular finance need. Your goal is to decide whether the single idea is similar enough to the idea(s) in the list that the single idea could be naturally included in the list. Since all the ideas are aimed at satisfying the above user need, that connection alone is NOT enough of a commonality to say Yes, there should be some other major semantic connection between the ideas. The ultimate goal is to avoid redundancy, so please be very liberal and do say Yes in any case where you think the singleton idea has some significant overlap in meaning with the idea(s) in the list; for example, you should definitely say Yes if the singleton idea shares a major key phrase with most or all of the list, provided that phrase does not trivially follow from the user interest in {items}.  You should give your answer by writing just Yes or No, Yes if it makes sense to include the single item in the list, No if it does not. Do not include any other information, or any explanation."

IDEA_CLASSIFY_MAIN_PROMPT_STR = "Decide whether or not the single idea can reasonably be included in the existing list of ideas. Output either Yes or No. Here is the list of ideas: {instance}. Here is the single idea: {label}"

IDEA_REDUNDANCY_PASS_MAIN_PROMPT_STR = "You are a expert financial analyst who has received a information request from an important client and is reviewing some initial brainstorming results provided by your underlings. They have independently reviewed documents and brainstormed some {ideas}, some of which have been automatically clustered together. You will review each cluster (which may consist of a single {idea} or multiple {ideas}), and for each cluster, decide whether it is a consistent, coherent custer with a single new mian idea that is appropriate, distinct answer to the question, or whether it involves a) different ideas that don't really belong together, particularly if you already have a cluster which already addresses one of these idea, or a cluster that otherwise does not form a single sensible answer to what the client asked (some of your underlings may have misinterpreted the client's need and it is important you do not let the client see their bad work!) or b) the cluster is coherent but redundant with a cluster you have already reviewed and the two must be collapsed into one. Specifically, you will proceed through the clusters in sequential order and for each cluster (again, there many be only one {idea} in your cluster), you will output the number of the cluster, and then, after a tab, output one of three possible options:\n1. if the cluster is a good one, a short 1-3 word label (no conjunctions, do not use words in the {idea} definition) that gives the core idea of the cluster, one that has no redundancy with those you have reviewed thus far\n2. if the cluster is incoherent, partially redundant, or otherwise a bad answer, output No\n3. If the cluster is completely redundant with an earlier cluster, output a number corresponding to an earlier cluster where you chose 1. That means that the two clusters can be collapsed into a single cluster and still fall under the provided label. If you output a number, that number must ALWAYS be a lower number than the current cluster. For example, you might output:\n1\tGenerative AI\n2\tNo\n3\t1\nThis indicates that 1 is a good cluster about Generative AI, 2 is a bad cluster, and 3 is redundant with 1 (it is also about Generative AI) and the two can be collapsed. Again, when indicating redundancy you may only output cluster numbers smaller than the current cluster number, and that cluster number must be one that you have already provided a label for. Though it most cases you will keep most clusters (never remove all or most of them!), it is very important that your {ideas} do not seem repetitive and that you collapse and/or remove anything that is similar, especially among the top 5, each of the top 5 ideas (the top 5 clusters listed, excluding those you remove or collapse) must be a very distinct idea with NO major semantic overlap with anything else in the top 5. If any of the initial {ideas} contained in a cluster refers to an idea you have already used as a label, you must never, ever label that cluster, but either combine it with an existing one (if the cluster as a whole can fall under that label) or say No and remove it. Do not output anything but lines with cluster numbers and your answer (i.e. {idea} title, No, or another cluster number) for each. Here is the specific {idea} definition of what you are identifying for the client: {idea_definition}. Here are the enumerated {idea} clusters, delimited by ---\n---\n{clusters}\n---\nNow iterate over the clusters and decide if the cluster is good, bad, or redundant:\n"

FINAL_IDEA_SYS_PROMPT_STR = (
    "You are a financial analyst reviewing a large body of text data to try to collect a set of specific {ideas} that will satisfy the particular needs of one of your clients. You have already finished the initial brainstorming round which you have farmed out to your subordinate analysts, and now your goal is to do a final write up describing and justifying the choice of one particular {idea} as one of the answers to the client's request. You will be provided with the brainstorming need (the {idea} definition) of the client and one or more initial {ideas} from the brainstorming session. If you are given multiple {ideas} they must be collapsed into one coherent {idea}; the {idea} should include as much of the individual parts as provided while not seeming to just be a simple agglomeration of two or more separate {ideas}; pick one key main idea, and absolutely avoid using the word 'and' in your title! If there is only one {idea} provided, you may use it more or less directly, subject to the restrictions on wording below. Regardless, your output must be laser-focused on the {idea} or {ideas} you are given, do not talk about any other {idea} that might be relevant to the client's need. The chat context may discuss finding stocks relevant to your {ideas}, but you must absolutely avoid mention of any specific companies in your {idea} title unless that company is mentioned directly in the {idea} definition (you may of course mention specific stocks as examples in your longer description).  Your output will consist first of a 3 to 10 word title that clearly expresses the idea (the title alone must contain all major information needed to understand what the idea is, be specific!) and is clearly an instance of the specific {idea} definition provided. I repeat, the idea title must NEVER, EVER have a company name in it unless that company name appears in the {idea} definition, even if the company is referred to in the initial {ideas} you are drawing from. Then, on the next line, write a text of at most 500 words which describes the {idea} and grounds it in evidence from the source documents. All the source documents provided should have at least some relevance to the {idea} you are looking at. You should try to include as many citations in your description as you reasonably can, and you must absolutely cite every source document if there are less than 5, or at least 5 if there are more than 5. The more citations, the better, as long as you can include them coherently! Never, ever just cite a single document, and try to avoid avoid citing only documents relevant to a single company unless that company is mentioned in the need expressed by the user; diversity is very important. Note that you should generally avoid directly making direct reference to `source documents`. Also, avoid repeating the {idea} definition directly in either your title or description, except when it is not possible to interpret the meaning of the {idea} without it. Do not add any markdown to your title!\n"
    + CITATION_PROMPT
)

FINAL_IDEA_MAIN_PROMPT_STR = (
    "Do a final write-up proposing one particular {idea} as a good answer to a user brainstorming need. Here is the specific definition of the {ideas} required by the user: {idea_definition}. Here is the {idea} or {ideas} from the initial brainstorming that you are focusing on here:\n---\n{initial_ideas}\n---\nHere are the source document you will pull information from and cite:\n---\n{source_documents}\n---\n"
    + CITATION_REMINDER
    + "Now write your final {idea}, with title, main description, and citations:\n"
)

FINAL_IDEA_SYS_PROMPT = Prompt(FINAL_IDEA_SYS_PROMPT_STR, "FINAL_IDEA_SYS_PROMPT")

FINAL_IDEA_MAIN_PROMPT = Prompt(FINAL_IDEA_MAIN_PROMPT_STR, "FINAL_IDEA_MAIN_PROMPT")

IDEA_NOUN_MAIN_PROMPT_STR = 'Give the following description of something a user of your platform is interested in, output the singular and plural form of the noun (a single word) that can be used to refer to these things of interest. Note that usually the plural form is already part of the description. For example, if the description is `President Biden\'s top policies`, you would output the following json: `{{"singular":"policy", "plural":"policies"}}`. You should output only a json with these two keys (`singular` and `plural`) and exactly one word for the value of each, do not output anything else. Here is the description: {idea_definition}. Now output your json:\n'

IDEA_NOUN_MAIN_PROMPT = Prompt(IDEA_NOUN_MAIN_PROMPT_STR, "IDEA_NOUN_MAIN_PROMPT")

BRAINSTORM_IDEAS_DESCRIPTION = "This tool takes a description of some set of items that the client is looking for (which we will refer to here as ideas), and a list of texts to search though to come up those ideas. Examples of kinds of `ideas` might include patterns, themes, topics, trends, events, policies, initiatives, conditions, etc., basically anything that tends to be listed that can be expressed in text. Note that the client will nearly always use use a more specific word than idea, you should basically use this tool whenever the client says they want come up with a list of X from some texts. However, this tool must never be used for identifying 'investment ideas' i.e. if the client seems to be looking directly for a list of stocks, in such cases you must use the recommendation tool and/or one of the stock filtering or stock ranking tools. This tool should also only be used when it is clear that the client wishes to extract these ideas by brainstorming them from text data. This tool MUST be used when the client clearly want multiple ideas where they intend to do some kind of further analysis on each idea (e.g. finding stocks affected by each theme derived from a brainstorming session). The idea_description should be a noun phrase that describes the particular ideas the user is looking for, e.g. 'macroeconomic patterns'. Note that in many cases the requirements of the client are very specific, and you must include all relevant details in the idea_definition passed to this tool; the tool cannot read the client's input nor the the plan you have written, so make sure the idea description provides all information provided by the client relevent to the ideas they wish to generate. The tool will NOT have the context to understand vague idea_descriptions! For example, if the client wants key initiatives spearheaded by Biden during his presidency, your idea_description must be something like 'President Biden's key initiatives', leaving out mention of Biden or the presidency in the idea_description would be a serious mistake,  mentioning it elsewhere in the plan is not a substitue! If the client mentions a specific number of ideas they want, you must include that integer as the max_ideas argument. If you intend to call per_idea_summarize_output on the list of ideas produced by this tool, you must never, ever call prepare_output on the ideas before the call to per_idea_summarize_output, you must only call prepare_output once on any given list of ideas!"
