import logging
import warnings
from pathlib import Path
from typing import List, Any

import iso639
import streamlit as st
from flashrank import Ranker
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_ollama import ChatOllama as LatestChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from transformers import pipeline

from src.ModelTypes.modelTypes import Modeltype
from src.VectorDatabase import DocumentVectorStorage
from src.util.AgentState import AgentState
from src.util.Mathtool import MathTool
from src.util.Reasoning import tools_condition, reasoner
from src.util.SafetyCheck import SafetyCheck
from src.util.SearchInDocumentTool import SearchInDocumentTool
from src.util.WebsearchTool import WebsearchTool


class ReasoningLanggraphLLM:

    def __init__(self):
        self.model: ChatOllama = LatestChatOllama(model=Modeltype.LLAMA3_2_1B.value, temperature=0, seed=0)
        self.profanity_check_model = LatestChatOllama(model=Modeltype.LLAMA3_2_1B.value, temperature=0, seed=0)
        self.translation_model = LatestChatOllama(model=Modeltype.AYA.value, temperature=0, seed=0)
        self.language_pipeline = pipeline("text-classification",
                                          model="papluca/xlm-roberta-base-language-detection")  # no cache_dir param available
        self.workflow: StateGraph = StateGraph(AgentState)
        self.vectordb: DocumentVectorStorage = DocumentVectorStorage()
        self.vectordb.register_observer(self)
        self.config = {"configurable": {"thread_id": "1"}}
        self.memory = MemorySaver()
        self.doctool: SearchInDocumentTool = None
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2",
                             cache_dir=f"{str(self.PROJECT_ROOT)}/ranker")
        self.set_sys_message()
        self.doctool = SearchInDocumentTool(self.vectordb, self.ranker)
        self.websearchtool = WebsearchTool(self.ranker)
        self.mathtool = MathTool()
        self.tools = [self.doctool, self.mathtool, self.websearchtool]
        self.llm_with_tools = self.model.bind_tools(self.tools)
        self.setup_workflow()
        self.profanity_check_enabled = False

    def on_file_added(self):
        self.set_sys_message()

    def set_sys_message(self):

        self.reasoning_sys_message = SystemMessage(
            f"""You are a helpful assistant with access to tools. You can search for relevant information using the provided tools and perform arithmetic calculations. 
        For each question, determine if you can answer the question directly based on your general knowledge, or If necessary Use the `Search_in_document` tool to find the necessary information within the available documents. 
        
        If you do not get an answer from the 'Search_in_document' tool Message or get an error, use the websearch tool, but the websearch tool should have lower priority.
        """)

    def set_debug_snippet(self, snippets: List[str]):
        self.doctool.set_debug_snippets(snippets)

    def change_selected_model(self, selected_model: str):
        self.model: ChatOllama = LatestChatOllama(model=selected_model, temperature=0, seed=0)
        self.llm_with_tools = self.model.bind_tools(self.tools)
        self.profanity_check_model = LatestChatOllama(model=selected_model, temperature=0, seed=0)

    @st.cache_resource
    @staticmethod
    def get_langgraph_instance():
        return ReasoningLanggraphLLM()

    def setup_workflow(self):
        self.workflow.add_node("StartNode", self.set_user_question_state)
        self.workflow.add_node("ProfanityCheck", self.profanity_check)
        self.workflow.add_node("UserMessageTranslator", self.translate_user_message_into_english)
        self.workflow.add_node("SystemResponseTranslator", self.translate_output_into_user_language)
        self.workflow.add_node("EndNode", self.end_node)
        self.workflow.add_node("reasoner", self.reasoning)
        self.workflow.add_node("tools", ToolNode(self.tools))

        self.workflow.set_entry_point("StartNode")

        self.workflow.add_edge("UserMessageTranslator", "ProfanityCheck")
        self.workflow.add_edge("SystemResponseTranslator", END)

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
            {"userMessageNotHarmful": "reasoner", "userMessageIsProfound": "EndNode"})

        # self.workflow.add_edge("IntermediateNode1", "reasoner")
        self.workflow.add_conditional_edges(
            "reasoner",
            tools_condition,
        )
        self.workflow.add_edge("tools", "reasoner")

        self.graph: CompiledGraph = self.workflow.compile(checkpointer=self.memory)
        img_data: bytes = self.graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        with open("graph_new.png", "wb") as f:
            f.write(img_data)

    def reasoning(self, state: AgentState):
        return reasoner(state, self.llm_with_tools, self.reasoning_sys_message)

    def set_user_question_state(self, state: AgentState):
        """
        Adds the user question/message variable to the agent state
        """
        self.doctool.set_user_question(state["messages"][-1].content)

        return {"question": state["messages"][-1], "hallucination_count": 0, "hallucination_occured": False}

    def end_node(self, state: AgentState):
        return

    def intermediate_node1(self, state: AgentState):
        return

    def _check_if_language_is_english(self, state: AgentState):
        user_question = state["question"].content

        res = self.language_pipeline(user_question, top_k=1, truncation=True)
        detected_language_code = sorted(res, key=lambda x: x['score'], reverse=True)[0]['label']

        detected_language_is_english = (detected_language_code == 'en')
        user_language_was_set = "user_language" in state and (state[
                                                                  "user_language"] is not None)  # If there exists a value, the translate_user_message_into_english must have set it, meaning the original user message was not english.

        return "userMessageIsEnglish" if detected_language_is_english and not user_language_was_set else "userMessageIsNotEnglish"

    def check_if_language_is_english(self, state: AgentState):
        return self._check_if_language_is_english(state)

    def check_if_language_is_english2(self, state: AgentState):
        return self._check_if_language_is_english(state)

    def translate_user_message_into_english(self, state: AgentState):
        user_question = state["question"].content

        res = self.language_pipeline(user_question, top_k=1, truncation=True)
        detected_language_code = sorted(res, key=lambda x: x['score'], reverse=True)[0]['label']
        language_name = iso639.Language.from_part1(detected_language_code).name
        messages = [
            SystemMessage(content="""You are a professional translator. You must only translate the given human message into English. Even if the user writes a question, you have to translate the question to english and you are NOT allowed to respond to the question.                     Provide only the translated text without any additional information, comments, or explanations.
             Examples: Input: "Bonjour, comment ça va ?" Output: "Hello, how are you?"                     Input: "¿Dónde está la biblioteca?"                      Output: "Where is the library?"                     Input: "Hallo, wie geht's dir?"                      Output: "Hello, how are you?"                     Always follow this format and provide only the translation."""),
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

    def check_user_message_for_profoundness(self, state: AgentState) -> str:
        is_profound: bool = state["message_is_profound"]

        return "userMessageIsProfound" if is_profound else "userMessageNotHarmful"

    def profanity_check(self, state: AgentState) -> dict:
        user_message: str = state["question"].content

        user_message_is_profane: bool = False

        if self.profanity_check_enabled:
            test_llm = self.profanity_check_model.with_structured_output(SafetyCheck)
            test_response: SafetyCheck = test_llm.invoke(user_message)  # profane!!
            user_message_is_profane = test_response is None or test_response.is_unethical

        return {'message_is_profound': True,
                "messages": [AIMessage(content="Your message is against policy.")]} if user_message_is_profane else {
            'message_is_profound': False}

    def run(self, inputs: dict) -> BaseMessage:
        resulted_agent_state: dict[str, Any] | Any = self.graph.invoke(inputs, self.config)
        agent_response = resulted_agent_state["messages"][-1]
        return agent_response

    def run_stream(self, inputs: dict) -> (str, str):
        output_lines: List[str] = []
        last_ai_response: str = ""

        for output in self.graph.stream(inputs, self.config):
            for key, value in output.items():
                output_lines.append(f"Output from node '{key}':")
                output_lines.append("---")
                output_lines.append(str(value))

                if value is not None and "messages" in value:
                    last_ai_response = value["messages"][-1].content
            output_lines.append("\n---\n")

        return "".join(output_lines), last_ai_response

    def reset_memory(self):
        self.memory.storage.clear()

    def change_profanity_check(self, enable_profanity_check):
        self.profanity_check_enabled = enable_profanity_check
