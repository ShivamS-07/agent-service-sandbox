# flake8: noqa
from agent_service.utils.prompt_utils import Prompt

# shared plan stuff
# , and, in parenthesis after each one, a function you think you will use to include it your plan

PLAN_RULES = """- Your output script must be valid Python code
- The first line of your code should be a comment starting with "# Must haves: " where you should briefly list the key elements of the user input that you must include in your plan in short phrases of 1-3 words.
- All non-comment lines of code must consist of one assignment statement
- The left side of the assignment should be a single new variable
- The right side of your assignment statement must consist of exactly one function call, multiple function calls must use multiple lines, you must not output line that does not contain a tool call.
- I will say it again, because it is VERY IMPORTANT: each line of your code must have exactly one function call in it, no more, and no less. A functional call within the arguments, e.g. func1(a=func2(b)) counts as a second function for this purpose: DO NOT DO IT
- It is equally bad to have no function calls, you absolutely must not use a line to define a list (e.g. a = [b,c,d]) or a string (e.g. a = "a string"), just put the list/string directly in the arguments of any functions that need it
- The function used must come from the provided list of tools, you must not use any other standard Python functions
- You must not use any Python operators other than the assignment operator, you must the provided functions to carry out any required operations
- In particular, you absolutely must not use the Python indexes [], e.g. you must not write anything involving the characters `[]` that is not a definiing a list. This includes inside a function argument, e.g. func(a=b[0]) is not allowed. Instead use the provided function which does the same thing.
- Any arguments to the function must match the provided function headers in type
- You must include all required arguments (those without defaults)
- If values for any of the optional arguments (those with defaults) are mentioned or implied by the user input include those optional arguments, unless it is already the default, at which point you must leave it out.
- Use explicit keyword arguments for all your arguments
- Any arguments to the function must consist of a single string, integer, float, or boolean literal, a variable, or a list of literals or variables. Do not include more complex python data structures, and you must never include a function as part of an argument. If you need to modify a existing variable in any way other that creating a list containing it, put it on a separate line
- You must NOT create a Python list on a separate line without a function call in it since each line MUST have one function. You must absolutely NOT produce a line like a = [b, c, d]. Instead, you should create the list in the argument, i.e. `a = func(arg=[a, b, c])`
- If you write a string literal in an argument, it must be delimited with double quotes and the string literal must not contain any of the following five characters delimited by `: `=,()"`
- Before each line of code, you will have a comment line which first has the one function you will use, and then each of the arguments to the function which must follow all the requirements, i.e. not containing functions, indexes, or operators. If there is not one function, or one of the argument violates one of the above requirements, write another comment line which fixes the problems before you write the next line of code
- After each assignment, on the same line, write a comment that explains what is happening in that step. It should be fully understandable to a non-coder who cannot read the code itself
- In your comment, if the function name does not mention identifiers, you should not refer to them either, just talk about stocks, not stock indentifiers
- The final variable assigned in the last step will correspond to the intended output which best satisfies the user's information need. You do not need to do anything further with that variable, just assign it like any other step
- You must output a script/plan. If you are unable to write a script which fully satisfies the user's information need with the provided functions, or the exact user needs are unclear, write the best script that you can given the limitations
- You must output only one script/plan, do not output multiple plans, pick the best one using the provided guidelines
- You must write nothing other than the script
"""

PLAN_GUIDELINES = """The top priority of a plan is that the information needs of the user are being fully satisfied. Otherwise, please consider the following when selecting the best plan:
- short plans are preferred over long ones
- in particular, it is better to use functions that do everything you need to do in one batched call, rather than separate calls, and it is very, very bad to do both versions redundantly in one script
- simple functions are preferred over API calls
- internal APIs calls are preferred over external API
- If you can accomplish the same thing with or with an LLM, do it without the LLM, it's much cheaper
- Please pay special attention to whether you might need to apply filtering functions to your data, many applications require filtering based on the specific needs of the user
- You must never assume what the date is, you must use a tool that knows the date
- Your lines can be as long as needed, do not violate any of the rules to avoid having long lines
- Before you write a line, consider what each of the arguments of your function will be; if any of them is in the wrong form and so requires a function to be called on them before you can use it as a proper argument to this function, do that first!"""

PLAN_EXAMPLE = "if you had the following two functions if your function set:\ndef add(num1: float, num2: float) -> float:\n# This function adds two numbers\ndef multiply(num1:float, num2:float) -> float:\n# this function multiplies two numbers\nAnd if the user message was:\nAdd 2.4 and 7.93 and multiply the result by 3\nThen you would output:\n # Must haves: add 2.4 and 7.93, multiply by 3.0\n# add 2.4 7.93\nsum = add(num1=2.4, num2=7.93)   # Add 2.4 and 7.93\n# product sum 3.0\nproduct = multiply(num1=sum, num2=3.0)  # Multiply that sum by 3"


# initial planner

PLANNER_SYS_PROMPT_STR = "You are a financial data analyst. Your main goal will be to write a plan to satisfy the provided information need of a client, in the form of a short Python script. You will be provided with categorized groups of functions you are allowed to call in your program, with a description of each group, and then a Python header and a description for each function in the group. You must adhere to a very specific format for your output, if you do not satisfy all of the rules your output will be rejected. Here are the rules you must follow:\n{rules}\n\n Here are some further guidelines that will help you in making good choices while writing your script/plan:\n{guidelines}\n\n Here is a example:\n{example}. Here are the functions/tools available:\n{tools}"

PLANNER_MAIN_PROMPT_STR = "Write a simple Python script that uses only functions in the provided list to satisfy the information need expressed in the client message. Please be very careful of the rules, the most important being that each line of your script must call exactly one function from the provided list, if you do not follow that rule, the script will fail and your work is worthless. Here is the client message:\n---\n{message}\n---\n.\n Now write your python script:\n"


### Dataclasses

PLANNER_SYS_PROMPT = Prompt(PLANNER_SYS_PROMPT_STR, "PLANNER_SYS_PROMPT")
PLANNER_MAIN_PROMPT = Prompt(PLANNER_MAIN_PROMPT_STR, "PLANNER_MAIN_PROMPT")
