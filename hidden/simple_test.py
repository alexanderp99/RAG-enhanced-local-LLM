from langchain_community.chat_models import ChatOllama

model = ChatOllama(model='llama3:instruct')

"""
This code is now continued in "LanggraphLLM.py
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

from langgraph.prebuilt import ToolExecutor

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tools = [WikipediaQueryRun(api_wrapper=api_wrapper)]
tool_executor = ToolExecutor(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    rag_context: str


def should_continue(state):
    if True:
        return "end"
    else:
        return "continue"


def call_model(state):
    return


def call_tool(state):
    return


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

workflow.add_edge('action', 'agent')

app = workflow.compile()

inputs = {"messages": [HumanMessage(content="WHy is the sky blue?")]}
print(app.invoke(inputs))
