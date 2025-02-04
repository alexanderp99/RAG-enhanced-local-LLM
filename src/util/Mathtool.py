import logging
from typing import Optional, Type

import numexpr as ne
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from numpy import ndarray
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class MathtoolInput(BaseModel):
    expression: str = Field(
        description="The numerical expression to evaluate.")


class MathTool(BaseTool):
    name: str = "Math Tool"
    description: str = "Calculates a numerical expression"
    args_schema: Type[BaseModel] = MathtoolInput
    return_direct: bool = False

    def __init__(self):
        super().__init__()

    def _run(self, expression: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> ndarray:
        logger.info(f"Expression query: {expression}")

        result = ne.evaluate(expression)
        logger.info(F"Expression result: {result}")
        return ne.evaluate(expression)
