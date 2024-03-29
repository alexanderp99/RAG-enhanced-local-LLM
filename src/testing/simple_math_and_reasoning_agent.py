from langchain.agents import load_tools, Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchResults

# TUTORIAL: https://towardsdatascience.com/building-a-math-application-with-langchain-agents-23919d09a4d3


model = Ollama(model="gemma:2b")

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
    tools=[DuckDuckGoSearchResults(), math_tool, word_problem_tool],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

print(agent.invoke(
    {
        "input": "What is 5+5?"}))

######
"""
Langgraph documentation

graph.add_node("oracle", model)
graph.add_edge("oracle", END)

graph.set_entry_point("oracle")

runnable = graph.compile()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named {name} who always speaks in pirate dialect"),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model
"""
