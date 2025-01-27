import logging
from typing import Optional, Any, Type

from duckduckgo_search import DDGS
from flashrank import Ranker, RerankRequest
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class WebsearchInput(BaseModel):
    query: str = Field(
        description="A single(!) relevant query for websearch", )


class WebsearchTool(BaseTool):
    name: str = "Search the web"
    description: str = "Search the web using a single(!) query"
    args_schema: Type[BaseModel] = WebsearchInput
    return_direct: bool = False
    ranker: Optional[Any] = None

    def __init__(self, ranker):
        super().__init__()
        self.ranker = ranker

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        logger.info(f"Searchquery: {query}")

        return "Websearch could not return value"

        """results = DDGS().text(query, max_results=5)

        rerank_request = RerankRequest(query=query, passages=[{"text": result["body"]} for result in results])
        ranked_results = self.ranker.rerank(rerank_request)
        ranked_results_sorted = sorted(ranked_results, key=lambda x: x["score"], reverse=True)

        logger.info(f"Searchquery response: {ranked_results_sorted[0]["text"]}")

        return ranked_results_sorted[0]["text"]"""

        # logger.info(f"Searchquery response: {snippet_concat}")
        # return snippet_concat
