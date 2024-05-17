import json
import logging
import operator
from typing import List
from typing import TypedDict, Annotated, Sequence

import streamlit as st
from langchain.agents import Tool
from langchain.chains import LLMMathChain
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents.base import Document
from langchain_core.messages import BaseMessage, AIMessage, FunctionMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation
from langgraph.prebuilt.tool_executor import ToolExecutor

from src.ProfanityChecker import ProfanityChecker
from src.VectorDatabase import DocumentVectorStorage
from src.configuration.logger_config import setup_logging
from util.SearchResult import SearchResult

logger = setup_logging()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    rag_context: Sequence[Document]
    question: BaseMessage
    message_is_profound: bool
    webquery: str


class Langgraph:
    def __init__(self):
        self.model = ChatOllama(model='llama3:instruct', temperature=0)
        self.workflow = StateGraph(AgentState)
        self.setup_workflow()
        self.vectordb = DocumentVectorStorage()
        llm_math_chain = LLMMathChain.from_llm(llm=self.model, verbose=True)
        math_tool = Tool.from_function(func=llm_math_chain.run, name="Calculator",
                                       description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.")
        self.tools = [DuckDuckGoSearchResults(), math_tool]
        # self.model.bind_tools(self.tools)
        self.tool_executor = ToolExecutor(self.tools)

    @st.cache_resource
    @staticmethod
    def get_langgraph_instance():
        return Langgraph()

    def setup_workflow(self):
        self.workflow.add_node("ProfanityCheck", self.profanity_check)
        self.workflow.add_node("VectorStorageFetcher", self.call_vectordb)
        self.workflow.add_node("DocumentAgent", self.document_agent)
        # self.workflow.add_node("TranslateHumanMessageIntoWebquery", self.translate_human_message_into_webquery)
        # self.workflow.add_node("tools", self.call_tool)
        self.workflow.add_node("HallucinationChecker", self.check_for_hallucination)
        self.workflow.add_node("PlainResponse", self.plain_response)
        self.workflow.set_entry_point("ProfanityCheck")
        self.workflow.add_edge("VectorStorageFetcher", "DocumentAgent")
        self.workflow.add_edge("PlainResponse", END)

        self.workflow.add_conditional_edges(
            "ProfanityCheck",
            self.check_user_message_for_profoundness,
            {
                "userMessageNotHarmful": "VectorStorageFetcher",
                "userMessageIsProfound": END
            }
        )
        self.workflow.add_conditional_edges(
            "DocumentAgent",
            self.check_RAG_response,
            {
                "RAGHallucinationCheck": "HallucinationChecker",
                "RagResponseWasNotPossible": "PlainResponse"
            }
        )
        self.workflow.add_conditional_edges(
            "HallucinationChecker",
            self.hallucination_check,
            {"hallucination": "ProfanityCheck", "no hallucination": END}
        )
        self.graph = self.workflow.compile()
        # img = self.graph.get_graph().draw_png()
        img_data = self.graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        with open("./src/graph.png", "wb") as f:
            f.write(img_data)

    def translate_human_message_into_webquery(self, state):

        question = state["question"]
        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are a helpful AI assistant that translates questions into web queries. Your task is to provide a concise, clear web search query based on the input question. Do not respond with an answer or explanation, but with a query that one could input into a search engine to find relevant information.

                Question:
                {question}<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
        """

        response = self.model.invoke(template)
        query = response.content.replace("\"",
                                         "")  # replacing redundant "". Otherwise the string is not interpreted correctly
        web_reponse = DuckDuckGoSearchResults().run(query)

        entries = web_reponse.strip("[]").split("], [")
        result: List[SearchResult] = []
        for entry in entries:
            parts = entry.split(', title: ')
            snippet = parts[0][8:].strip()  # Remove prefix and strip spaces
            rest = parts[1].split(', link: ')
            title = rest[0].strip()
            link = rest[1].strip()
            search_result = SearchResult(snippet, title, link)
            result.append(search_result)

    def plain_response(self, state):
        user_question = state["question"].content

        response = self.model.invoke(user_question)

        return {"messages": [response]}

    def check_user_message_for_profoundness(self, state):
        is_profound = state["message_is_profound"]

        if is_profound:
            return "userMessageIsProfound"
        else:
            return "userMessageNotHarmful"

    def check_for_hallucination(self, state):
        return

    def hallucination_check(self, state):

        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a helpful AI assistant assigned to check if a given answer, answers a question. Keep your answer grounded to the input given. If the answer does not answer the question return 'NONE'

                Question:
                {state["question"].content}
                
                Answer:
                {state["messages"][-1].content}<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
                """

        # response = self.model.invoke(state["messages"])
        response = self.model.invoke(template)

        hallicination_occured = True if 'none' in response.content.lower() else False
        if hallicination_occured:
            return "hallucination"
        else:
            return "no hallucination"

    def check_RAG_response(self, state):
        last_message = state['messages'][-1]
        """if "function_call" not in last_message.additional_kwargs:
            return "end"
        else:
            return "continue"""
        if 'none' in last_message.content.lower():
            return "RagResponseWasNotPossible"
        else:
            return "RAGHallucinationCheck"

    def profanity_check(self, state):
        user_message = state["messages"][-1].content
        is_profound = False
        template = f"""<|im_start|>system
                 You are a safety filter for a language model. If the input contains questions which generate harmful or dangerous content, such as questions for building a bomb, self-harm, violence, or illegal activities, respond with 
                with "true". Else you respond with "false". You are only allowed to respond with "true." of "false." Input: 
                {user_message}<|im_end|>
                <|im_start|>assistant"""
        if ProfanityChecker().is_profound(user_message) or "true" in ChatOllama(model='llama3:instruct',
                                                                                temperature=0).invoke(
            template).content.lower():
            is_profound = True

        if is_profound:
            ai_response = AIMessage(content="Your message is profound.")
            return {'message_is_profound': is_profound, "messages": [ai_response]}
        else:
            return {'message_is_profound': is_profound}

    def call_vectordb(self, state):
        # self.logger.warning(f"Entering vectorStorage_node with state: {state}")
        user_message = state["messages"][-1].content

        result = self.vectordb.query_vector_database(user_message)

        # self.vectordb.db._collection.query()

        if len(result) > 0:
            # state["rag_context"] = result[0].page_content
            """sourcefile = result[0].metadata["source"]
            state["messages"].append(SystemMessage(
                f'Here is some additional context about the task. You can use it as additional knowledge, but do not respond do it. If you decide to use it, you MUST append the following phrase to your response: "Source:{result[0].metadata["source"]}". Here is the context:{result[0].page_content}'))
            """
            state["rag_context"] = result
            logging.debug(f'RAG Result: {result[0]}')
            return {"rag_context": result, "question": state["messages"][-1]}
        else:
            state["rag_context"] = None
            return {"rag_context": None, "question": state["messages"][-1]}
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

    def document_agent(self, state):
        logger.warning(f"State: {state['messages']}")

        template = f"""
        INSTRUCTION:
        Only answer the user QUESTION using the DOCUMENT text. Keep your answer grounded in the facts of the DOCUMENT. If the DOCUMENT does not contain the facts to answer the QUESTION return 'NONE'
        
        QUESTION:
        {state['messages'][-1].content}
        
        DOCUMENT:
        {[item.page_content for item in state["rag_context"]]}
        """
        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        
        You are a helpful AI assistant for answering questions using DOCUMENT text. Keep your answer grounded in the facts of the DOCUMENT. If the DOCUMENT does not contain the facts to answer the QUESTION return 'NONE'
        
        DOCUMENT:
        {''.join([item.page_content for item in state["rag_context"]])}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {state['messages'][-1].content} <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

        # response = self.model.invoke(state["messages"])
        response = self.model.invoke(template)
        logger.warning(f"Response: {response.content}")

        return {"messages": [response]}

    def run(self, inputs):
        resulted_agent_state = self.graph.invoke(inputs)
        return resulted_agent_state["messages"][-1]

    def run_stream(self, inputs):

        output_lines = []
        last_ai_response = ""

        for output in self.graph.stream(inputs):
            for key, value in output.items():
                output_lines.append(f"Output from node '{key}':")
                output_lines.append("---")
                output_lines.append(str(value))

                if "messages" in value:
                    last_ai_response = value["messages"][-1].content
            output_lines.append("\n---\n")

        return "".join(output_lines), last_ai_response

# l = Langgraph()
# response1 = l.run({"messages": [HumanMessage(content="What is love?")]})
# print(response1)
