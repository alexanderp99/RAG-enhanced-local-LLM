import logging
from pathlib import Path
from typing import List, Optional, Any, Type

from flashrank import RerankRequest
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize
from DocumentWrapper import DocumentWrapper

from util import logger


class DocumentInput(BaseModel):
    queries: List[str] = Field(
        description="Generate multiple related queries based on:1. Synonyms for key terms.2. Questions asking about the same topic.3. Common expressions for searching similar concepts.")


class SearchInDocumentTool(BaseTool):
    name: str = "Search_in_document"
    description: str = "Searches for content within documents using multiple queries for better results."
    args_schema: Type[BaseModel] = DocumentInput
    return_direct: bool = False
    vectordb: Optional[Any] = None
    ranker: Optional[Any] = None
    user_question: str = " "
    model: Optional[Any] = None
    snippets: Optional[Any] = None

    def set_user_question(self, question):
        self.user_question = question

    def __init__(self, database, ranker):
        super().__init__()
        self.vectordb = database
        self.ranker = ranker

    def set_debug_snippets(self, snippets: List[str]):
        self.snippets = snippets

    def _run(self, queries: List[str], run_manager: Optional[CallbackManagerForToolRun] = None) -> str:

        queries.append(self.user_question)
        logger.info(f"Document Queries: {queries}")

        try:
            unique_docs = set()
            unique_docs.update(self.vectordb.query_vector_database_with_keywords(queries))

            for query in queries:
                result = self.vectordb.query_vector_database(query)
                for doc in result:
                    converted_doc = DocumentWrapper(doc)
                    unique_docs.add(converted_doc)

            rerank_request = RerankRequest(query=self.user_question,
                                           passages=[{"text": each_doc.page_content} for each_doc in unique_docs])
            ranked_results = self.ranker.rerank(rerank_request)
            ranked_results = sorted(ranked_results, key=lambda x: x["score"], reverse=True)

            some_snippet_in_result = any(
                each_snippet.replace(" ", "").replace(",", "") in each_ranked_result["text"].replace("\n", "").replace(
                    " ", "").replace(",", "")
                for each_ranked_result in ranked_results
                for each_snippet in self.snippets
            )

            logger.info(f"Some_snippet_in_result: {some_snippet_in_result}")

            docs = [Document(page_content=res["text"]) for res in ranked_results][:2]

        except Exception as e:
            docs = [Document(page_content="There was no content found")]

        all_docs_string = f''.join(
            [f"Context {idx}: " + item.page_content.replace("\n", "") + "\n\n" for idx, item in enumerate(docs)])

        logger.info(f"Database Context: {all_docs_string}")

        summaries: List[str] = self.vectordb.get_document_summaries()
        for i, each_summary in enumerate(summaries):
            summaries[i] = each_summary.replace("The document", "The context")
        all_docs_string = "Metainfo:" + "".join(summaries) + all_docs_string

        return all_docs_string
