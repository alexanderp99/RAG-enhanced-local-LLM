from logging import Logger
from typing import List

from profanity_check import predict

from src.configuration.logger_config import setup_logging

logger: Logger = setup_logging()


class ProfanityChecker:
    @staticmethod
    def is_profound(phrase: str) -> List[int]:
        return predict([phrase])
