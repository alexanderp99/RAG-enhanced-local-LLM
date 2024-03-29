from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults

# Source:https://python.langchain.com/docs/integrations/tools/ddg


model = ChatOllama(model="gemma")
search = DuckDuckGoSearchResults()
tools = load_tools(["ddg-search"], llm=model)
agent = initialize_agent(tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
print(agent.run("what is latest news in indian crickted"))
