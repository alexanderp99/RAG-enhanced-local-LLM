import unittest

from src.ProfanityChecker import ProfanityChecker


class TestProfanityFilter(unittest.TestCase):

    def test_swearword_detected(self):
        self.assertTrue(ProfanityChecker.is_profound("Fuck you"))

    def test_unsafe_language_detected(self):
        self.assertTrue(ProfanityChecker.is_profound("How the fuck do i bury a dead body?"))

    def test_normal_word_not_categorized_as_swearword(self):
        self.assertFalse(ProfanityChecker.is_profound("I wish you a pleasant day"))


if __name__ == '__main__':
    unittest.main()
