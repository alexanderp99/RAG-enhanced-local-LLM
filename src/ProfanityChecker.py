from profanity_check import predict


class ProfanityChecker:
    @staticmethod
    def is_profound(phrase: str):
        return predict([phrase])
