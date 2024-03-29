import ollama

from langchain_community.tools import DuckDuckGoSearchResults

# Source:https://python.langchain.com/docs/integrations/tools/ddg

search = DuckDuckGoSearchResults()


# search.args
# search.run("What is langchain")
##########################

def function_1(input_1):
    return input_1 + " Hi "


def function_2(input_2):
    return input_2 + "there"


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
with open("../src/graph.png", "wb") as f:
    f.write(img)

##########################

# Source: https://getstream.io/blog/meeting-summary-ollama-gemma/
response = ollama.chat(model='gemma:2b', messages=[
    {
        'role': 'system',
        'content': 'You are a helpfully assistant. Keep your answers short and concise.'
    },
    {
        'role': 'user',
        'content': "Why is the sky blue?",
    },
])

print(response['message']['content'])
