import unittest

from langchain_core.messages import HumanMessage

from src.LanggraphLLM import Langgraph


class TestLanggraph(unittest.TestCase):

    def setUp(self):
        self.agent = Langgraph()

    def test_should_continue_with_profanity(self):
        result = self.agent.check_user_message_for_profoundness({'messages': [HumanMessage(content='Fuck you')]})
        self.assertEqual(result, 'end')

    def test_should_continue_without_profanity(self):
        result = self.agent.check_user_message_for_profoundness({'messages': [HumanMessage(content='hello world')]})
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

    def test_mathematical_ability(self):
        inputs = {"messages": [HumanMessage(content="What is 30 times 230?")]}
        result = self.agent.run(inputs)
        print("hi")

    def test_mathematical_ability2(self):
        inputs = {"messages": [HumanMessage(content="What is 33.4 times 230.7?")]}
        result = self.agent.run_stream(inputs)
        print("hi")

    def test_websearch_ability(self):
        inputs = {
            "messages": [HumanMessage(content="Search on the web what Phi3 by microsoft is and tell me what it is")]}
        result = self.agent.run(inputs)
        print("hi")


if __name__ == '__main__':
    unittest.main()
