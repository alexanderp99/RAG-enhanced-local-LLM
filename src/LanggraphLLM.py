import json
import logging
import operator
from typing import TypedDict, Annotated, Sequence

from langchain.agents import Tool
from langchain.chains import LLMMathChain
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation
from langgraph.prebuilt.tool_executor import ToolExecutor

from src.ProfanityChecker import ProfanityChecker
from src.VectorDatabase import DocumentVectorStorage
from src.configuration.logger_config import setup_logging

logger = setup_logging()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    rag_context: str


class Langgraph:
    def __init__(self):
        self.model = ChatOllama(model='phi3')
        self.workflow = StateGraph(AgentState)
        self.setup_workflow()
        self.vectordb = DocumentVectorStorage()
        llm_math_chain = LLMMathChain.from_llm(llm=self.model, verbose=True)
        math_tool = Tool.from_function(func=llm_math_chain.run, name="Calculator",
                                       description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.")
        self.tools = [DuckDuckGoSearchResults(), math_tool]
        # self.model.bind_tools(self.tools)
        self.tool_executor = ToolExecutor(self.tools)

    def setup_workflow(self):
        self.workflow.add_node("ProfanityCheck", self.call_model)
        self.workflow.add_node("VectorStorageFetcher", self.call_vectordb)
        self.workflow.add_node("Agent", self.agent_node)
        self.workflow.add_node("tools", self.call_tool)
        self.workflow.set_entry_point("ProfanityCheck")
        self.workflow.add_edge("VectorStorageFetcher", "Agent")
        self.workflow.add_edge("Agent", END)
        self.workflow.add_edge("tools", "Agent")
        self.workflow.add_conditional_edges(
            "ProfanityCheck",
            self.should_continue,
            {
                "continue": "VectorStorageFetcher",
                "end": END
            }
        )
        self.workflow.add_conditional_edges(
            "Agent",
            self.should_continue2,
            {
                "continue": "tools",
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

    def should_continue2(self, state):
        messages = state['messages']
        last_message = messages[-1]
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        else:
            return "continue"

    def call_model(self, state):
        # self.logger.warning("call model")
        return

    def call_vectordb(self, state):
        # self.logger.warning(f"Entering vectorStorage_node with state: {state}")
        user_message = state["messages"][-1].content

        result = self.vectordb.query_vector_database(user_message)
        if len(result) > 0:
            state["rag_context"] = result[0].page_content
            state["messages"].append(SystemMessage(
                f'Here is some additional context about the task. You can use it as additional knowledge, but do not respond do it:{result[0].page_content}'))
            logging.debug(f'RAG Result: {result[0].page_content}')
        else:
            state["rag_context"] = None
        return

    def call_tool(self, state):
        messages = state['messages']
        last_message = messages[-1]
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
        )
        logger.debug(f"The agent action is {action}")
        response = self.tool_executor.invoke(action)
        logger.debug(f"The tool result is: {response}")
        function_message = FunctionMessage(content=str(response), name=action.tool)
        return {"messages": [function_message]}

    def agent_node(self, state):
        logger.warning(f"State: {state['messages']}")
        response = self.model.invoke(state["messages"])
        logger.warning(f"Response: {response.content}")
        return {"messages": [response]}

    def run(self, inputs):
        resulted_agent_state = self.graph.invoke(inputs)
        return resulted_agent_state["messages"][-1]

    def run_stream(self, inputs):
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                logger.debug(f"Output from node '{key}':")
                logger.debug("---")
                logger.debug(value)
            logger.debug("\n---\n")

# l = Langgraph()
# response1 = l.run({"messages": [HumanMessage(content="What is love?")]})
# print(response1)
