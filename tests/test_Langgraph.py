import unittest

from langchain_core.messages import HumanMessage

from src.LanggraphLLM import Langgraph


class TestLanggraph(unittest.TestCase):

    def setUp(self):
        self.agent = Langgraph()

    def test_should_continue_with_profanity(self):
        result = self.agent.should_continue({'messages': [HumanMessage(content='Fuck you')]})
        self.assertEqual(result, 'end')

    def test_should_continue_without_profanity(self):
        result = self.agent.should_continue({'messages': [HumanMessage(content='hello world')]})
        self.assertEqual(result, 'continue')

    def test_call_model(self):
        test_state = {}
        self.assertIsNone(self.agent.call_model(test_state))
        self.assertDictEqual(test_state, {})

    def test_run(self):
        inputs = {"messages": [HumanMessage(content="What is the color of the sky?")]}
        result = self.agent.run(inputs)
        self.assertTrue("shorter wavelengths of light (like blue and violet) are scattered more evenly throughout the "
                        "atmosphere by tiny molecules of gases like nitrogen and oxygen." in result)


if __name__ == '__main__':
    unittest.main()
