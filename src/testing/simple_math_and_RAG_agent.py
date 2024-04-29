from langchain_community.llms import Ollama

# TUTORIAL: https://towardsdatascience.com/building-a-math-application-with-langchain-agents-23919d09a4d3


model = Ollama(model="gemma:2b")
from langchain_community.tools import DuckDuckGoSearchResults
from src.VectorDatabase import DocumentVectorStorage

vectordb = DocumentVectorStorage()

tools = [DuckDuckGoSearchResults(), vectordb.get_local_knowledge]
from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages
