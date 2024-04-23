import logging
import operator
from typing import TypedDict, Annotated, Sequence

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from src.ProfanityChecker import ProfanityChecker
from src.VectorDatabase import DocumentVectorStorage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    rag_context: str


class Langgraph:
    def __init__(self):
        self.model = ChatOllama(model='llama3:instruct')
        self.logger = logging.getLogger()
        self.workflow = StateGraph(AgentState)
        self.setup_workflow()
        self.vectordb = DocumentVectorStorage()

    def setup_workflow(self):
        self.workflow.add_node("ProfanityCheck", self.call_model)
        self.workflow.add_node("VectorStorageFetcher", self.call_tool)
        self.workflow.add_node("Agent", self.agent_node)
        self.workflow.set_entry_point("ProfanityCheck")
        self.workflow.add_edge("VectorStorageFetcher", "Agent")
        self.workflow.add_edge("Agent", END)
        self.workflow.add_conditional_edges(
            "ProfanityCheck",
            self.should_continue,
            {
                "continue": "VectorStorageFetcher",
                "end": END
            }
        )
        self.graph = self.workflow.compile()

    def should_continue(self, state):
        user_message = state["messages"][-1].content
        if ProfanityChecker().is_profound(user_message):
            ai_response = AIMessage(content="Your message is profound.")
            state["messages"].append(ai_response)
            return "end"
        else:
            return "continue"

    def call_model(self, state):
        self.logger.warning("call model")
        return

    def call_tool(self, state):
        self.logger.warning(f"Entering vectorStorage_node with state: {state}")
        user_message = state["messages"][-1].content

        result = self.vectordb.query_vector_database(user_message)
        if len(result) > 0:
            state["rag_context"] = result[0].page_content
            state["messages"].append(HumanMessage(result[0].page_content))
            # logging.warning(f'RAG Result: {result[0].page_content}')
        else:
            state["rag_context"] = None
        self.logger.warning(f"Exiting vectorStorage_node with state: {state}")
        return

    def agent_node(self, state):
        self.logger.warning(f"State: {state['messages']}")
        response = self.model.invoke(state["messages"])
        self.logger.warning(f"Response: {response}")
        return

    def run(self, inputs):
        return self.graph.invoke(inputs)

    def run_stream(self, inputs):
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
            print("\n---\n")
