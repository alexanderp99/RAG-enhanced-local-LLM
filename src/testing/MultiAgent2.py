from langchain_community.llms import Ollama

from src.VectorDatabase import DocumentVectorStorage

# Source: https://python.langchain.com/docs/integrations/llms/ollama
llm = Ollama(model="gemma:2b")
# llm.invoke("Tell me a joke")


from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import ToolExecutor

vectordb = DocumentVectorStorage()

tools = [DuckDuckGoSearchResults(), vectordb.get_local_knowledge]
tool_executor = ToolExecutor(tools)

from langgraph.graph import Graph

# Define a Langchain graph
workflow = Graph()

workflow.add_node("node_1", function_1)
workflow.add_node("node_2", function_2)

workflow.add_edge('node_1', 'node_2')

workflow.set_entry_point("node_1")
workflow.set_finish_point("node_2")

app = workflow.compile()
app.invoke("Hello")

img = app.get_graph().draw_png()
with open("../graph.png", "wb") as f:
    f.write(img)
