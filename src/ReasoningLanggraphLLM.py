import logging
import operator
import warnings
from typing import List, Any, Optional, Annotated, TypedDict, Union, Literal, Sequence

import streamlit as st
from flashrank import Ranker, RerankRequest
# from ftlangdetect import detect
from iso639 import Lang
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents.base import Document
from langchain_core.messages import BaseMessage, AIMessage, AnyMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama as LatestChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from transformers import pipeline

from modelTypes import Modeltype
from src.VectorDatabase import DocumentVectorStorage
from util.AgentState import AgentState
from util.SearchResult import SearchResult as WebSearchResult, SearchResult
from langchain_core.tools import tool, BaseTool

logger = logging.getLogger(__name__)


class SafetyCheck(BaseModel):
    """Checks if a given input contains inhumane, illegal, or unethical content."""

    input: str = Field(description="The phrase, question, or demand to be classified.")
    is_unethical: bool = Field(
        default=False,
        description="True if the input contains unethical content. Unethical content includes, but is not limited to, mentions of violence, self-harm, illegal activity, lawbreaking, harm or danger to others, terrorism, or anything intended to cause injury or suffering."
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


class QuestionAnswer(BaseModel):
    """Gives an answer to a question using Context."""

    answer: str = Field(
        description="The answer that should answer the question using the context"
    )


class SearchInDocumentTool(BaseTool):
    name: str = "search_in_document"
    description: str = "Searches for content within documents."
    return_direct: bool = False
    vectordb: Optional[Any] = None

    def __init__(self, database):
        super().__init__()
        self.vectordb = database

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        result: list[Document] = self.vectordb.query_vector_database(query)

        rerankrequest = RerankRequest(query=query, passages=[{
            "text": doc.page_content}
            for doc in result])
        ranked_results = self.ranker.rerank(rerankrequest)

        search_result_idx = 0

        ranked_results = [sorted(ranked_results, key=lambda x: x["score"], reverse=True)[search_result_idx]]
        docs = [Document(page_content=res["text"]) for res in ranked_results]

        return docs[0]


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


def tools_condition(
        state: Union[list[AnyMessage], dict[str, Any], BaseModel],
        messages_key: str = "messages",
) -> Literal["tools", "EndNode"]:
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
    return "EndNode"


class GraphState(TypedDict):
    """State of the graph."""
    query: str
    finance: str
    final_answer: str
    # intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    messages: Annotated[list[AnyMessage], operator.add]

    rag_context: Sequence[Document]
    question: HumanMessage
    message_is_profound: bool
    webquery: str
    web_results: List[SearchResult]
    user_language: str
    hallucination_occured: bool
    hallucination_count: int
    whole_available_rag_context: Sequence[Document]


class ReasoningLanggraphLLM:

    def __init__(self):
        self.model: ChatOllama = LatestChatOllama(model=Modeltype.LLAMA3_1_8B.value, temperature=0)
        self.profanity_check_model = LatestChatOllama(model='llama3.1:latest', temperature=0)
        self.translation_model = LatestChatOllama(model="aya", temperature=0)
        self.workflow: StateGraph = StateGraph(AgentState)
        self.vectordb: DocumentVectorStorage = DocumentVectorStorage()
        self.allow_document_search = True
        self.allow_profanity_check = True
        self.allow_hallucination_check = True
        self.config = {"configurable": {"thread_id": "1"}}
        self.memory = MemorySaver()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2",
                                 cache_dir="/ranker")  # # if not sys.warnoptions:  # flashrank\Ranker.py:115: ResourceWarning: unclosed file <_io.TextIOWrapper name='\\ranker\\ms-marco-MiniLM-L-12-v2\\config.json' mode='r' encoding='cp1252'> config = json.load(open(str(self.model_dir / "config.json"))) https://stackoverflow.com/questions/26563711/disabling-python-3-2-resourcewarning
        # has to come last
        logger.debug("Langgraph CTOR 1!")
        logger.debug("Langgraph CTOR 2!")
        logger.debug("Langgraph CTOR 3!")
        self.setup_workflow()

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
        self.workflow.add_node("IntermediateNode1", self.intermediate_node1)
        # new

        doctool = SearchInDocumentTool(self.vectordb)
        tools = [add, multiply, divide, doctool]
        self.llm_with_tools = self.model.bind_tools(tools)

        sys_message = SystemMessage(
            """You are a helpful assistant with access to tools. You can search for relevant information using the provided tools and perform arithmetic calculations. 
        For each question, determine if you can answer the question directly based on your general knowledge, or If necessary Use the `search_in_document` tool to find the necessary information within the available documents.""")

        def reasoner(state):
            # query = state["query"]
            messages = state["messages"]
            # System message
            no_human_message_added = not (any([isinstance(each_message, HumanMessage) for each_message in messages]))
            sys_msg = sys_message
            # message = human_message
            message = state["question"]
            if no_human_message_added:
                messages.append(message)
            result = [self.llm_with_tools.invoke([sys_msg] + messages)]
            return {"messages": result}

        self.workflow.add_node("reasoner", reasoner)
        self.workflow.add_node("tools", ToolNode(tools))

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
            {"userMessageNotHarmful": "IntermediateNode1", "userMessageIsProfound": "EndNode"})

        # new
        self.workflow.add_edge("IntermediateNode1", "reasoner")
        self.workflow.add_conditional_edges(
            "reasoner",
            tools_condition,
        )
        self.workflow.add_edge("tools", "reasoner")

        self.graph: CompiledGraph = self.workflow.compile(checkpointer=self.memory)
        img_data: bytes = self.graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        # with open("src/graph.png", "wb") as f:
        #    f.write(img_data)

    def set_user_question_state(self, state: AgentState):
        """
        Adds the user question/message variable to the agent state
        """
        return {"question": state["messages"][-1], "hallucination_count": 0, "hallucination_occured": False}

    def end_node(self, state: AgentState):
        return

    def intermediate_node1(self, state: AgentState):
        return

    def check_if_document_search_enabled(self, state: AgentState):
        return "DocumentSearchEnabled" if self.allow_document_search else "DocumentSearchDisabled"

    def check_if_vector_storage_fetcher_returned_result(self, state: AgentState):
        return "no_result" if state["rag_context"] is None else "multiple_results"

    def _check_if_language_is_english(self, state: AgentState):
        user_question = state["question"].content

        # detected_language_code = detect(user_question)
        # detected_language_code = detect(text=user_question, low_memory=False)["lang"]

        model_ckpt = "papluca/xlm-roberta-base-language-detection"
        pipe = pipeline("text-classification",
                        model=model_ckpt)  # no , cache_dir="text_language_classification/ available
        res = pipe(user_question, top_k=1, truncation=True)
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

        model_ckpt = "papluca/xlm-roberta-base-language-detection"
        pipe = pipeline("text-classification",
                        model=model_ckpt)  # no , cache_dir="text_language_classification/ available
        res = pipe(user_question, top_k=1, truncation=True)
        detected_language_code = sorted(res, key=lambda x: x['score'], reverse=True)[0]['label']
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
        user_message: str = state["question"]

        messages = [
            SystemMessage(
                content=f"You are a helpful AI assistant that translates questions into web queries. Your task is to provide a concise, clear web search query based on the input question. Do not respond with an answer or explanation, but with a query that one could input into a search engine to find relevant information."),
            HumanMessage(content=question.content)
        ]

        # test_llm = self.profanity_check_model.with_structured_output(WebSearch)
        # test_response: SafetyCheck = test_llm.invoke(f"Question: {user_message}")

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

        messages = [
            SystemMessage(
                content=f"""You are a helpful AI assistant for answering questions using DOCUMENT text. Keep your answer grounded in the facts of the DOCUMENT. If the DOCUMENT does not contain the facts to answer the QUESTION return 'NONE'. If you use the document text, mention the link from the source.

                DOCUMENT:
                {str(web_results)}"""),
            HumanMessage(content=human_message.content)
        ]

        response: BaseMessage = self.model.invoke(messages)

        return {"messages": [response]}

    def check_user_message_for_profoundness(self, state: AgentState) -> str:
        is_profound: bool = state["message_is_profound"]

        return "userMessageIsProfound" if is_profound else "userMessageNotHarmful"

    def check_for_hallucination(self, state: AgentState) -> None:

        hallicination_occured: bool = None
        hallucination_count: int = state["hallucination_count"] if state["hallucination_count"] is not None else 0

        if self.allow_hallucination_check:
            messages = [
                SystemMessage(
                    content=f"""You are a helpful AI assistant assigned to check if a given response, answers a question. Keep your answer grounded to the input given. If the answer does not answer the question return 'NONE'

                            Response:
                            {state["question"].content}"""),
                HumanMessage(content=state["messages"][-1].content)
            ]
            response: BaseMessage = self.model.invoke(messages)

            if 'none' in response.content.lower():
                hallucination_count += 1
                hallicination_occured = True
            else:
                hallicination_occured = False
        else:
            hallicination_occured = False

        return {"hallicination_occured": hallicination_occured, "hallucination_count": hallucination_count}

    def hallucination_check(self, state: AgentState) -> str:

        if self.allow_hallucination_check:
            hallicination_occured: bool = state["hallucination_occured"]
        else:
            hallicination_occured = False

        return "hallucination" if hallicination_occured else "no hallucination"

    def check_RAG_response(self, state: AgentState) -> str:
        last_message: BaseMessage = state['messages'][-1]
        rag_context_size = len(state["whole_available_rag_context"])

        document_search_no_more_possible = 'none' in last_message.content.lower() and (
                rag_context_size == state["hallucination_count"])
        answer_not_possible_but_more_context_available = 'none' in last_message.content.lower() and (
                rag_context_size > state["hallucination_count"])

        if document_search_no_more_possible:
            return "DocumentSearchNoMorePossible"
        elif answer_not_possible_but_more_context_available:
            return "RetryRagGeneration"
        else:
            return "RagCheckSuccessful"

    def profanity_check(self, state: AgentState) -> dict:
        user_message: str = state["question"].content

        user_message_is_profane: bool = None

        if self.allow_profanity_check:
            test_llm = self.profanity_check_model.with_structured_output(SafetyCheck)
            test_response: SafetyCheck = test_llm.invoke(user_message)  # profane!!
            user_message_is_profane = test_response is None or test_response.is_unethical
        else:
            user_message_is_profane = False

        return {'message_is_profound': True,
                "messages": [AIMessage(content="Your message is profane.")]} if user_message_is_profane else {
            'message_is_profound': False}

    def call_vectordb(self, state: AgentState) -> dict:
        user_message: str = state["question"].content
        hallucination_count: int = state["hallucination_count"]

        logger.debug(f'RAG Result: {docs[0]}' if docs else 'RAG Result: None')
        return {"rag_context": docs, "whole_available_rag_context": result}

    def document_agent(self, state: AgentState) -> dict:
        messages = [
            SystemMessage(
                content=f"""You are a helpful AI assistant for answering questions using DOCUMENT text. Keep your answer grounded in the facts of the DOCUMENT. If the DOCUMENT does not contain the facts to answer the QUESTION return 'NONE'
        
        DOCUMENT:
        {''.join([item.page_content for item in state["rag_context"]])}"""),
            HumanMessage(content=state['messages'][-1].content)
        ]

        response: BaseMessage = self.model.invoke(messages)

        # user_message = state['messages'][-1].content
        # rag_context = ''.join([item.page_content for item in state["rag_context"]])
        # test_llm = self.profanity_check_model.with_structured_output(
        #    QuestionAnswered)
        # test_response: SafetyCheck = test_llm.invoke(f"question:{user_message} \n Source:{rag_context}")

        hallucination_count: int = state["hallucination_count"]
        if 'none' in response.content.lower():
            hallucination_count += 1

        return {"messages": [response], "hallucination_count": hallucination_count}

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
