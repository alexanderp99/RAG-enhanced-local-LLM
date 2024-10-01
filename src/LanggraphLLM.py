import logging
import os
from logging import Logger
from typing import List, Any

import streamlit as st
from iso639 import Lang
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents.base import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_ollama import ChatOllama as LatestChatOllama
from langdetect import detect
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

from src.VectorDatabase import DocumentVectorStorage
from src.configuration.logger_config import setup_logging
from util.AgentState import AgentState
from util.WebSearchResult import WebSearchResult

logger: Logger = setup_logging()
HumanMessage

from pydantic import BaseModel, Field


class SafetyCheck(BaseModel):
    """Checks if a given phrase/question/demand is inhumane."""

    input: str = Field(description="The input which has to be classified")
    is_unethical: bool = Field(
        default=False,
        description="If the input is unethical or not. It is unethical if there is violence, self-harm, illegal, harmful, dangerous, hurts people, or terrorism mentioned."
    )


class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


class WebSearch(BaseModel):
    """Converts a question into a web query"""

    question: str = Field(description="The input question")
    web_query: str = Field(
        default=False,
        description="Concise and clear web search query based on the input question. A query that one could input into a search engine to find relevant information."
    )


class QuestionAnswered(BaseModel):
    """Checks if a given Source answers a question."""

    question: str = Field(description="The question")
    source: str = Field(description="The source which should answer the question")
    source_answers_question: bool = Field(
        default=True,
        description="If the source does not answer the question, this is False"
    )


class Langgraph:

    def __init__(self):
        self.model: ChatOllama = ChatOllama(model='llama3:instruct', temperature=0)
        self.profanity_check_model = LatestChatOllama(model='llama3.1', temperature=0)
        self.translation_model = ChatOllama(model="aya", temperature=0)
        self.workflow: StateGraph = StateGraph(AgentState)
        self.vectordb: DocumentVectorStorage = DocumentVectorStorage()
        self.setup_workflow()
        self.allow_document_search = False

    @st.cache_resource
    @staticmethod
    def get_langgraph_instance():
        return Langgraph()

    def setup_workflow(self):
        self.workflow.add_node("StartNode", self.set_user_question_state)
        self.workflow.add_node("ProfanityCheck", self.profanity_check)
        self.workflow.add_node("VectorStorageFetcher", self.call_vectordb)
        self.workflow.add_node("DocumentAgent", self.document_agent)
        self.workflow.add_node("RetrieveWebKnowledge", self.retrieve_web_knowledge)
        self.workflow.add_node("HallucinationChecker", self.check_for_hallucination)
        self.workflow.add_node("PlainResponse", self.plain_response)
        self.workflow.add_node("UserMessageTranslator", self.translate_user_message_into_english)
        self.workflow.add_node("SystemResponseTranslator", self.translate_output_into_user_language)
        self.workflow.add_node("EndNode", self.end_node)
        self.workflow.add_node("IntermediateNode1", self.intermediate_node1)

        self.workflow.set_entry_point("StartNode")

        self.workflow.add_edge("VectorStorageFetcher", "DocumentAgent")
        self.workflow.add_edge("PlainResponse", "EndNode")
        self.workflow.add_edge("RetrieveWebKnowledge", "PlainResponse")
        self.workflow.add_edge("UserMessageTranslator", "ProfanityCheck")
        self.workflow.add_edge("SystemResponseTranslator", END)

        self.workflow.add_conditional_edges("IntermediateNode1", self.check_if_document_search_enabled,
                                            {"DocumentSearchEnabled": "VectorStorageFetcher",
                                             "DocumentSearchDisabled": "RetrieveWebKnowledge"})
        self.workflow.add_conditional_edges(
            "EndNode", self.check_if_language_is_english2,
            {"userMessageIsEnglish": END, "userMessageIsNotEnglish": "SystemResponseTranslator"})
        self.workflow.add_conditional_edges(
            "StartNode",
            self.check_if_language_is_english,
            {"userMessageIsEnglish": "ProfanityCheck", "userMessageIsNotEnglish": "UserMessageTranslator"})
        self.workflow.add_conditional_edges(
            "ProfanityCheck",
            self.check_user_message_for_profoundness,
            {"userMessageNotHarmful": "IntermediateNode1", "userMessageIsProfound": "EndNode"})
        self.workflow.add_conditional_edges(
            "DocumentAgent",
            self.check_RAG_response,
            {"RAGHallucinationCheck": "HallucinationChecker",
             "RagResponseWasNotPossible": "RetrieveWebKnowledge"})
        self.workflow.add_conditional_edges(
            "HallucinationChecker",
            self.hallucination_check,
            {"hallucination": "ProfanityCheck", "no hallucination": "EndNode"})
        self.graph: CompiledGraph = self.workflow.compile()
        img_data: bytes = self.graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        file_path: str = os.path.join("src", "graph.png")
        with open(file_path, "wb") as f:
            f.write(img_data)

    def set_user_question_state(self, state: AgentState):
        """
        Adds the user question/message variable to the agent state
        """
        return {"question": state["messages"][-1]}

    def end_node(self, state: AgentState):
        return

    def intermediate_node1(self, state: AgentState):
        return

    def check_if_document_search_enabled(self, state: AgentState):
        if self.allow_document_search:
            return "DocumentSearchEnabled"
        else:
            return "DocumentSearchDisabled"

    def _check_if_language_is_english(self, state: AgentState):
        user_question = state["question"].content

        detected_language_code = detect(user_question)
        detected_language_is_english = (detect(detected_language_code) == 'en')

        user_language_was_set = "user_language" in state and (state[
                                                                  "user_language"] is not None)  # If there exists a value, the translate_user_message_into_english must have set it, meaning the original user message was not english.

        if detected_language_is_english and not user_language_was_set:
            return "userMessageIsEnglish"
        else:
            return "userMessageIsNotEnglish"

    def check_if_language_is_english(self, state: AgentState):
        return self._check_if_language_is_english(state)

    def check_if_language_is_english2(self, state: AgentState):
        return self._check_if_language_is_english(state)

    def translate_user_message_into_english(self, state: AgentState):
        user_question = state["question"].content

        detected_language_code = detect(user_question)
        language_name = Lang(detected_language_code).name

        messages = [
            SystemMessage(
                content="You are a professional translator. Your job is to translate the user input into english. Only respond with the translated sentence."),
            state["question"]
        ]

        response = self.translation_model.invoke(messages)
        return {"user_language": language_name, "question": HumanMessage(content=response.content)}

    def translate_output_into_user_language(self, state: AgentState):
        last_message: BaseMessage = state['messages'][-1]
        user_language: str = state["user_language"]

        # Here the AIMessage is transformed into a HumanMessage, because according to the aya modelfile, the model supports no 'assistant' role. Only 'user' role.
        messages = [
            SystemMessage(
                content=f"You are a professional translator. Your job is to translate the user message into {user_language}. Only respond with the translated sentence in {user_language}."),
            HumanMessage(content=last_message.content)
        ]

        response = self.translation_model.invoke(messages)
        return {"messages": [response]}

    def retrieve_web_knowledge(self, state: AgentState) -> dict:

        question: BaseMessage = state["question"]
        template: str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are a helpful AI assistant that translates questions into web queries. Your task is to provide a concise, clear web search query based on the input question. Do not respond with an answer or explanation, but with a query that one could input into a search engine to find relevant information.

                Question:
                {question}<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
        """
        messages = [
            SystemMessage(
                content=f"You are a helpful AI assistant that translates questions into web queries. Your task is to provide a concise, clear web search query based on the input question. Do not respond with an answer or explanation, but with a query that one could input into a search engine to find relevant information."),
            HumanMessage(content=question.content)
        ]

        test_llm = self.profanity_check_model.with_structured_output(WebSearch(question=question))
        test_response: SafetyCheck = test_llm.invoke(user_message)

        response: BaseMessage = self.model.invoke(messages)

        # response: BaseMessage = self.model.invoke(template)
        query: str = response.content.replace("\"",
                                              "")  # replacing redundant "". Otherwise the string is not interpreted correctly
        web_reponse = DuckDuckGoSearchResults().run(query)

        entries = web_reponse.strip("[]").split("], [")
        result: List[WebSearchResult] = []
        for entry in entries:
            parts = entry.split(', title: ')
            snippet = parts[0][8:].strip()  # Remove prefix and strip spaces
            rest: str = parts[1].split(', link: ')
            title: str = rest[0].strip()
            link: str = rest[1].strip()
            search_result: WebSearchResult = WebSearchResult(snippet, title, link)
            result.append(search_result)

        return {"web_results": result}

    def plain_response(self, state: AgentState) -> dict:

        web_results: List[WebSearchResult] = state["web_results"]
        human_message: HumanMessage = next(
            filter(lambda x: type(x) is type(HumanMessage("")), list(reversed(state['messages']))))

        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are a helpful AI assistant for answering questions using DOCUMENT text. Keep your answer grounded in the facts of the DOCUMENT. If the DOCUMENT does not contain the facts to answer the QUESTION return 'NONE'. If you use the document text, mention the link from the source.

                DOCUMENT:
                {str(web_results)}<|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                {human_message.content} <|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
                """

        messages = [
            SystemMessage(
                content=f"""You are a helpful AI assistant for answering questions using DOCUMENT text. Keep your answer grounded in the facts of the DOCUMENT. If the DOCUMENT does not contain the facts to answer the QUESTION return 'NONE'. If you use the document text, mention the link from the source.

                DOCUMENT:
                {str(web_results)}"""),
            HumanMessage(content=human_message.content)
        ]

        response: BaseMessage = self.model.invoke(messages)

        # response: BaseMessage = self.model.invoke(template)

        return {"messages": [response]}

    def check_user_message_for_profoundness(self, state: AgentState) -> str:
        is_profound: bool = state["message_is_profound"]

        if is_profound:
            return "userMessageIsProfound"
        else:
            return "userMessageNotHarmful"

    def check_for_hallucination(self, state: AgentState) -> None:
        return

    def hallucination_check(self, state: AgentState) -> str:

        template: str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a helpful AI assistant assigned to check if a given answer, answers a question. Keep your answer grounded to the input given. If the answer does not answer the question return 'NONE'

                Question:
                {state["question"].content}
                
                Answer:
                {state["question"].content}<|eot_id|>
                <|start_header_id|>assistant<|end_header_id|>
                """

        messages = [
            SystemMessage(
                content=f"""You are a helpful AI assistant assigned to check if a given response, answers a question. Keep your answer grounded to the input given. If the answer does not answer the question return 'NONE'

                Response:
                {state["question"].content}"""),
            HumanMessage(content={state["question"].content})
        ]

        response: BaseMessage = self.model.invoke(messages)

        # response: BaseMessage = self.model.invoke(template)

        hallicination_occured: bool = True if 'none' in response.content.lower() else False
        if hallicination_occured:
            return "hallucination"
        else:
            return "no hallucination"

    def check_RAG_response(self, state: AgentState) -> str:
        last_message: BaseMessage = state['messages'][-1]
        """if "function_call" not in last_message.additional_kwargs:
            return "end"
        else:
            return "continue"""
        if 'none' in last_message.content.lower():
            return "RagResponseWasNotPossible"
        else:
            return "RAGHallucinationCheck"

    def profanity_check(self, state: AgentState) -> dict:
        user_message: str = state["question"].content

        test_llm = self.profanity_check_model.with_structured_output(SafetyCheck)
        test_response: SafetyCheck = test_llm.invoke(user_message)  # profane!!

        return {'message_is_profound': test_response.is_unethical,
                "messages": [AIMessage(content="Your message is profound.")]} if test_response.is_unethical else {
            'message_is_profound': test_response.is_unethical}

    def call_vectordb(self, state: AgentState) -> dict:
        user_message: str = state["question"].content

        result: list[Document] = self.vectordb.query_vector_database(user_message)
        rag_context = result if result else None

        logging.debug(f'RAG Result: {result[0]}' if result else 'RAG Result: None')
        return {"rag_context": rag_context}

    def document_agent(self, state: AgentState) -> dict:
        template: str = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        
        You are a helpful AI assistant for answering questions using DOCUMENT text. Keep your answer grounded in the facts of the DOCUMENT. If the DOCUMENT does not contain the facts to answer the QUESTION return 'NONE'
        
        DOCUMENT:
        {''.join([item.page_content for item in state["rag_context"]])}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {state['messages'][-1].content} <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

        messages = [
            SystemMessage(
                content=f"""You are a helpful AI assistant for answering questions using DOCUMENT text. Keep your answer grounded in the facts of the DOCUMENT. If the DOCUMENT does not contain the facts to answer the QUESTION return 'NONE'
        
        DOCUMENT:
        {''.join([item.page_content for item in state["rag_context"]])}"""),
            HumanMessage(content={state['messages'][-1].content})
        ]

        response: BaseMessage = self.model.invoke(messages)

        # response = self.model.invoke(state["messages"])
        # response: BaseMessage = self.model.invoke(template)

        #
        user_message = state['messages'][-1].content
        rag_context = ''.join([item.page_content for item in state["rag_context"]])
        test_llm = self.profanity_check_model.with_structured_output(
            QuestionAnswered(question=user_message, source=rag_context))
        test_response: SafetyCheck = test_llm.invoke(user_message)

        return {"messages": [response]}

    def run(self, inputs: dict) -> BaseMessage:
        resulted_agent_state: dict[str, Any] | Any = self.graph.invoke(inputs)
        agent_response = resulted_agent_state["messages"][-1]
        return agent_response

    def run_stream(self, inputs: dict) -> (str, str):
        output_lines: List[str] = []
        last_ai_response: str = ""

        for output in self.graph.stream(inputs):
            for key, value in output.items():
                output_lines.append(f"Output from node '{key}':")
                output_lines.append("---")
                output_lines.append(str(value))

                if value is not None and "messages" in value:
                    last_ai_response = value["messages"][-1].content
            output_lines.append("\n---\n")

        return "".join(output_lines), last_ai_response
