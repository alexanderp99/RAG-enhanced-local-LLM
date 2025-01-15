import logging
from typing import List, Optional, Any, Type

from flashrank import RerankRequest
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field
from transformers import T5Tokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)


class DocumentInput(BaseModel):
    queries: List[str] = Field(
        description="Generate multiple related queries based on:1. Synonyms for key terms.2. Questions asking about the same topic.3. Common expressions for searching similar concepts.")


class SearchInDocumentTool(BaseTool):
    name: str = "search_in_document"
    description: str = "Searches for content within documents using multiple queries for better results."
    args_schema: Type[BaseModel] = DocumentInput
    return_direct: bool = False
    vectordb: Optional[Any] = None
    ranker: Optional[Any] = None
    user_question: str = " "

    def set_user_question(self, question):
        self.user_question = question

    def __init__(self, database, ranker):
        super().__init__()
        self.vectordb = database
        self.ranker = ranker

    def _run(self, queries: List[str], run_manager: Optional[CallbackManagerForToolRun] = None) -> str:

        queries.append(self.user_question)
        logger.info(f"Document Queries: {queries}")

        docs_t5 = 0
        try:
            unique_docs = set()
            for query in queries:
                result = self.vectordb.query_vector_database(query)
                for doc in result:
                    unique_docs.add(doc.page_content)

            rerank_request = RerankRequest(query=", ".join(queries), passages=[{"text": text} for text in unique_docs])
            ranked_results = self.ranker.rerank(rerank_request)
            ranked_results = sorted(ranked_results, key=lambda x: x["score"], reverse=True)

            docs = [Document(page_content=res["text"]) for res in ranked_results][:1]
            docs_t5 = [Document(page_content=res["text"]) for res in ranked_results][:4]

        except Exception as e:
            docs = [Document(page_content="There was no content found")]

        all_docs_string = ''.join(
            [f"Context {idx}: " + item.page_content.replace("\n", "") + "\n\n" for idx, item in enumerate(docs)])
        all_docs_string = 'Context: '.join(
            [item.page_content.replace("\n", "\n") for idx, item in enumerate(docs)])
        all_docs_t5 = 'Context: '.join(
            [item.page_content.replace("\n", "\n") for idx, item in enumerate(docs_t5)])

        summary = ""

        try:
            model_name = "google/flan-t5-large"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)

            input_text = f"Please answer the question with the context. Question: {self.user_question} \n. Context: {all_docs_t5}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

            summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, length_penalty=1.8,
                                         early_stopping=False, repetition_penalty=1.5, do_sample=False)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(e)

        answer = "Answer: " + summary
        response = answer + " ." + all_docs_string

        logger.info(f"T5 answer: {answer}")
        logger.info(f"Database Context: {all_docs_string}")

        return response
