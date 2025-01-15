import logging
from typing import Optional, Any, Type

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
        snippet_concat = "There was no content found"
        logger.info(f"Searchquery: {query}")
        # snippet_concat = "The coalition agreements failed between SPÖ and ÖVP and NEOS. now övp and fpö are discussing terms"

        """
        results = DDGS().text(query, max_results=5)

        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2",
                        cache_dir="/ranker")
        rerank_request = RerankRequest(query=query, passages=[{"text": result["body"]} for result in results])
        ranked_results = ranker.rerank(rerank_request)
        ranked_results_sorted = sorted(ranked_results, key=lambda x: x["score"], reverse=True)

        return ranked_results_sorted[0]["text"]
        """
        logger.info(f"Searchquery response: {snippet_concat}")
        return snippet_concat
