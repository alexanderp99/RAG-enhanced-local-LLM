import operator
from typing import List
from typing import TypedDict, Annotated, Sequence

from langchain_core.documents.base import Document
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage

from .SearchResult import SearchResult


class AgentState(TypedDict):
    """
    State that is used within the agent, acting as global mutuable state

    Attributes:
        messages (Annotated[Sequence[BaseMessage], operator.add]): All previous messages (human and system)
        rag_context (Sequence[Document]): A sequence of documents that provide context for retrieval-augmented generation (RAG).
        question (HumanMessage): The human message containing the question posed by the user.
        message_is_profound (bool): A boolean indicating whether the message is considered profound.
        webquery (str): A string representing the web query to be executed.
        web_results (List[SearchResult]): A list of search results retrieved from the web query.
        user_language (str): A string indicating the language used by the user.
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
    rag_context: Sequence[Document]
    question: HumanMessage
    message_is_profound: bool
    webquery: str
    web_results: List[SearchResult]
    user_language: str
