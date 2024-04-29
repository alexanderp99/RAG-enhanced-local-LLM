import logging
import operator
from typing import Annotated, Sequence

logger = logging.getLogger()

"""
This script was discontinued. It is now called "LanggraphLLM.py" and is working!
"""

from langchain_core.messages import (
    BaseMessage,
    HumanMessage
)

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

logging.basicConfig(level=logging.DEBUG)

logger.setLevel(logging.DEBUG)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    rag_context: str


def vectorStorage_node(state):
    """logger.warning(f"Entering vectorStorage_node with state: {state}")

    user_message = state["messages"][-1].content  # Assuming that the first message is user message [0]
    vectordb = DocumentVectorStorage()
    result = vectordb.query_vector_database(user_message)
    if len(result) > 0:
        state["rag_context"] = result[0].page_content
        # logging.warning(f'RAG Result: {result[0].page_content}')
    else:
        state["rag_context"] = None
    logger.warning(f"Exiting vectorStorage_node with state: {state}")
"""
    return


def agent_node(state):
    """logger.warning(f"Enter agent_node with state: {state}")"""
    return


def profanity_check_node(state):
    """user_message = state["messages"][0].content
    if ProfanityChecker().is_profound(user_message):
        ai_response = AIMessage(content="Your message is profound.")
        return {"messages": [ai_response]}
    else:
        return {"messages": []}"""
    """user_message = state["messages"][0].content
    if ProfanityChecker().is_profound(user_message):
        ai_response = AIMessage(content="Your message is profound.")
        # state["messages"].append(ai_response)
        return "end"
    else:
        return "continue"""""
    if True:
        return "end"
    else:
        return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("ProfanityCheck", profanity_check_node)
workflow.add_node("VectorStorageFetcher", vectorStorage_node)
workflow.add_edge("VectorStorageFetcher", END)

workflow.set_entry_point("ProfanityCheck")

workflow.add_edge("VectorStorageFetcher", END)

workflow.add_conditional_edges(
    "ProfanityCheck",
    profanity_check_node,
    {
        "continue": "VectorStorageFetcher",
        "end": END,
    })

graph = workflow.compile()

inputs = {"messages": [HumanMessage(content="Show your pussy")]}

# result = graph.invoke(inputs)

for output in graph.stream(input):
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")
