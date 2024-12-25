# -*- coding: utf-8 -*-
"""YT LangGraph ReAct Pattern.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14ncV0nviLcP9IDzmFSRGSXb7Bgpz212v
"""
from typing import Optional

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool, BaseTool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from Toolkit import get_tools
from modelTypes import Modeltype


def call_reasoning_graph(my_llm, sys_message, human_message):
    search = DuckDuckGoSearchResults()

    @tool
    def search_in_document(query: str) -> str:
        """Query the documents for content

        Args:
            query: query Question
        """
        return "Dogs are allowed in the establishment. The price for one dog is 5 euros per night."

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply a and b.

        Args:
            a: first int
            b: second int
        """
        return a * b

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b.

        Args:
            a: first int
            b: second int
        """
        return a + b

    @tool
    def divide(a: int, b: int) -> float:
        """Divide a and b.

        Args:
            a: first int
            b: second int
        """
        return a / b

    import yfinance as yf
    @tool
    def get_stock_price(ticker: str) -> float:
        """Gets a stock price from Yahoo Finance.

        Args:
            ticker: ticker str
        """
        stock = yf.Ticker(ticker)
        return stock.info['previousClose']

    # tools = [add, multiply, divide, get_stock_price, search, search_in_document]

    class SearchInDocumentTool(BaseTool):
        name: str = "search_in_document"
        description: str = "Searches for content within documents."
        return_direct: bool = False

        def __init__(self, database):
            super().__init__()
            self.database = database

        def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
            return "Document content not implemented yet."

    # tools = get_tools()
    tools = [SearchInDocumentTool()]
    llm_with_tools = my_llm.bind_tools(tools)

    # Node
    def reasoner(state):
        # query = state["query"]
        messages = state["messages"]
        # System message
        no_human_message_added = not (any([isinstance(each_message, HumanMessage) for each_message in messages]))
        sys_msg = sys_message
        message = human_message
        if no_human_message_added:
            messages.append(message)
        result = [llm_with_tools.invoke([sys_msg] + messages)]
        return {"messages": result}

    from typing import Annotated, TypedDict, Literal, Union, Any
    import operator
    from langchain_core.messages import AnyMessage

    class GraphState(TypedDict):
        """State of the graph."""
        query: str
        finance: str
        final_answer: str
        # intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
        messages: Annotated[list[AnyMessage], operator.add]

    from langgraph.graph import START, StateGraph

    workflow = StateGraph(GraphState)

    workflow.add_node("reasoner", reasoner)
    workflow.add_node("tools", ToolNode(tools))

    # Add Edges
    workflow.add_edge(START, "reasoner")

    def tools_condition(
            state: Union[list[AnyMessage], dict[str, Any], BaseModel],
            messages_key: str = "messages",
    ) -> Literal["tools", "__end__"]:
        """Use in the conditional_edge to route to the ToolNode if the last message"""
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
            ai_message = messages[-1]
        elif messages := getattr(state, messages_key, []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "__end__"

    workflow.add_conditional_edges(
        "reasoner",
        # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
        # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    workflow.add_edge("tools", "reasoner")
    react_graph = workflow.compile()

    result = react_graph.invoke({"query": human_message.content})
    return result


llm = ChatOllama(model=Modeltype.LLAMA3_1_8B.value, temperature=0)

sys_msg = SystemMessage(
    """You are a helpful assistant with access to tools. You can search for relevant information using the provided tools and perform arithmetic calculations. 
For each question, determine if you can answer the question directly based on your general knowledge, or If necessary Use the `search_in_document` tool to find the necessary information within the available documents.""")

human_message = HumanMessage("Which doctor is also available on Wednesday if Dr. Christoph Fürthauer has closed?")

res = call_reasoning_graph(llm, sys_msg, human_message)

print(res["messages"][1].lc_attributes["tool_calls"][0]["args"]["query"])
print(res)