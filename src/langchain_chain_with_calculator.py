from langchain.agents import load_tools, Tool, initialize_agent
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import MessageGraph
from langgraph.prebuilt import ToolExecutor

model = Ollama(model="gemma:7b")

toolss = load_tools(["llm-math"], llm=model)  # LLM uses LLMMathChain

problem_chain = LLMMathChain.from_llm(llm=model)
math_tool = Tool.from_function(name="Calculator",
                               func=problem_chain.run,
                               description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.")

word_problem_template = """You are a reasoning agent tasked with solving 
the user's logic-based questions. Logically arrive at the solution, and be 
factual. In your answers, clearly detail the steps involved and give the 
final answer. Provide the response in bullet points. 
Question  {question} Answer"""

math_assistant_prompt = PromptTemplate(input_variables=["question"],
                                       template=word_problem_template
                                       )
word_problem_chain = LLMChain(llm=model,
                              prompt=math_assistant_prompt)
word_problem_tool = Tool.from_function(name="Reasoning Tool",
                                       func=word_problem_chain.run,
                                       description="Useful for when you need to answer logic-based/reasoning questions.")

agent = initialize_agent(
    tools=[DuckDuckGoSearchResults()],
    llm=model,
    verbose=True,
    handle_parsing_errors=True
)

print(agent.invoke(
    {
        "input": "What is 5+5?"}))
