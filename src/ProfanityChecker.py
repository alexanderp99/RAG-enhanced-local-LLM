from profanity_check import predict

from src.configuration.logger_config import setup_logging

logger = setup_logging()


class ProfanityChecker:
    @staticmethod
    def is_profound(phrase: str):
        return predict([phrase])
