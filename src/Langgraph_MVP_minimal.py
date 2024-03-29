import operator
from typing import Annotated, Sequence

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from src.LocalFileIndexer import DocumentVectorStorage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    rag_context: str


def vectorStorage_node(state):
    user_message = state["messages"][0].content  # Assuming that the first message is user message [0]
    query = "What is causal inference?"  # MAKE DYNAMIC
    vectordb = DocumentVectorStorage()
    result = vectordb.query_vector_database(query)
    if len(result) > 0:
        state["rag_context"] = result[0].page_content
    else:
        state["rag_context"] = None
    return state


def agent_node(state):
    pass
    return state


workflow = StateGraph(AgentState)

workflow.add_node("VectorStorageFetcher", vectorStorage_node)
workflow.add_node("Agent", agent_node)

workflow.add_edge('VectorStorageFetcher', 'Agent')

workflow.set_entry_point("VectorStorageFetcher")
workflow.set_finish_point("Agent")
graph = workflow.compile()

graph.invoke({
    "messages": [
        HumanMessage(
            content="Fetch the UK's GDP over the past 5 years,"
                    " then draw a line graph of it."
                    " Once you code it up, finish."
        )
    ],
})
