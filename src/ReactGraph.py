import logging
import operator
from typing import Annotated, TypedDict
from typing import Any

import streamlit as st
import yfinance as yf
# from ftlangdetect import detect
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, SystemMessage, AnyMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_ollama import ChatOllama as LatestChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode, tools_condition

from LanggraphLLM import Langgraph
from ModelTypes.modelTypes import Modeltype
from src.VectorDatabase import DocumentVectorStorage

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State of the graph."""
    query: str
    finance: str
    final_answer: str
    # intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    messages: Annotated[list[AnyMessage], operator.add]


class ReactLanggraph:

    def __init__(self):
        self.model: ChatOllama = LatestChatOllama(model=Modeltype.LLAMA3_1_8B.value, temperature=0)

        search = DuckDuckGoSearchRun()

        tools = [self.add, self.multiply, self.divide, search, self.get_stock_price]

        self.workflow: StateGraph = StateGraph(GraphState)

        self.workflow.add_node("start", self.set_state)
        self.workflow.add_node("reasoner", self.reasoner)
        self.workflow.add_node("tools", ToolNode(tools))

        self.workflow.set_entry_point("start")
        self.workflow.add_edge("start", "reasoner")
        self.workflow.add_conditional_edges(
            "reasoner",
            tools_condition,
        )
        self.workflow.add_edge("tools", "reasoner")

        self.llm_with_tools = self.model.bind_tools(tools)
        self.vectordb: DocumentVectorStorage = DocumentVectorStorage()
        self.config = {"configurable": {"thread_id": "1"}}
        self.memory = MemorySaver()
        logger.debug("Langgraph CTOR 1!")
        self.setup_workflow()

    @st.cache_resource
    @staticmethod
    def get_langgraph_instance():
        return Langgraph()

    def setup_workflow(self):
        self.graph: CompiledGraph = self.workflow.compile(checkpointer=self.memory)
        img_data: bytes = self.graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        with open("./ReactGraph.png", "wb") as f:
            f.write(img_data)

    def reasoner(self, state: MessagesState):
        messages = state["messages"]
        # System message
        sys_msg = SystemMessage(
            content="You are a helpful assistant tasked with using tools.")

        result = [self.llm_with_tools.invoke([sys_msg] + messages)]
        return {"messages": result}

    def set_state(self, state: MessagesState):
        return {"question": state["messages"][-1]}

    def multiply(a: int, b: int) -> int:
        """Multiply a and b.

        Args:
            a: first int
            b: second int
        """
        return a * b

    # This will be a tool
    def add(a: int, b: int) -> int:
        """Adds a and b.

        Args:
            a: first int
            b: second int
        """
        return a + b

    def divide(a: int, b: int) -> float:
        """Divide a and b.

        Args:
            a: first int
            b: second int
        """
        return a / b

    def get_stock_price(ticker: str) -> float:
        """Gets a stock price from Yahoo Finance.

        Args:
            ticker: ticker str
        """
        # """This is a tool for getting the price of a stock when passed a ticker symbol"""
        stock = yf.Ticker(ticker)
        return stock.info['previousClose']

    def run(self, inputs: dict) -> BaseMessage:
        resulted_agent_state: dict[str, Any] | Any = self.graph.invoke(inputs, self.config)
        agent_response = resulted_agent_state["messages"][-1]
        return agent_response
