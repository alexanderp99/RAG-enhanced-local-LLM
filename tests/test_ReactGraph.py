import unittest

from langchain_core.messages import HumanMessage

from src.ReactGraph import ReactLanggraph


class TestLanggraph(unittest.TestCase):

    def setUp(self):
        self.agent = ReactLanggraph()

    def test_should_continue_with_profanity(self):
        inputs = {"messages": [HumanMessage(content="What is the color of the sky?")]}
        result = self.agent.run(inputs)
        self.assertEqual(result, 'end')


if __name__ == '__main__':
    unittest.main()
