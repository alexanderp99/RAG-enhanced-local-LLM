from typing import Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SearchInDocumentTool(BaseTool):
    name: str = "search_in_document"
    description: str = "Searches for content within documents."
    return_direct: bool = False

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        # Simulate document search (replace with your actual logic)
        return "Document content not implemented yet."


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiplies two numbers."
    return_direct: bool = False

    class Configuration(BaseModel):
        a: int = Field(description="The first number")
        b: int = Field(description="The second number")

    args_schema: Type[BaseModel] = Configuration

    def _run(
            self,
            a: int,
            b: int,
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> int:
        return a * b


class AddTool(BaseTool):
    name: str = "add"
    description: str = "Adds two numbers."
    return_direct: bool = False

    class Configuration(BaseModel):
        a: int = Field(description="The first number")
        b: int = Field(description="The second number")

    args_schema: Type[BaseModel] = Configuration

    def _run(
            self,
            a: int,
            b: int,
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> int:
        return a + b


class DivideTool(BaseTool):
    name: str = "divide"
    description: str = "Divides two numbers."
    return_direct: bool = False

    class Configuration(BaseModel):
        a: int = Field(description="The first number")
        b: int = Field(description="The second number")

    args_schema: Type[BaseModel] = Configuration

    def _run(
            self,
            a: int,
            b: int,
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> float:
        return a / b


def get_tools():
    return [
        SearchInDocumentTool()
    ]
