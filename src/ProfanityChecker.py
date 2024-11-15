from typing import List

from profanity_check import predict


class ProfanityChecker:
    @staticmethod
    def is_profound(phrase: str) -> List[int]:
        return predict([phrase])
