import unittest

from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.LanggraphLLM import Langgraph
from src.VectorDatabase import DocumentVectorStorage


# Source:https://python.langchain.com/docs/integrations/tools/ddg

class QuestionAnsweredCheck(BaseModel):
    """Checks if a given question is answered sufficiently"""

    question: str = Field(description="The input question")
    fact: str = Field(description="A fact")
    answer: str = Field(
        description="The answer that should answer the question"
    )
    answered_sufficiently: bool = Field(
        desription="The check if the answer answers the question. The fact is used to verify that",
        default=False)


class TestVectorDatabase(unittest.TestCase):

    def setUp(self):
        self.agent: Langgraph = Langgraph()
        self.document_vector_storage: DocumentVectorStorage = self.agent.vectordb
        self.document_vector_storage.get_indexed_filenames()
        self.chatmodel: ChatOllama = ChatOllama(model='llama3:instruct', temperature=0)

    def test_querying_the_internet_works(self):
        search = DuckDuckGoSearchResults()
        web_result = search.run("What is langchain")
        expected_result = "LangChain is a powerful tool that can be used to build a wide range of LLM-powered applications"
        self.assertTrue(expected_result in web_result)

    def _check_single_question_answer_ability(self, question, fact):
        inputs = {"messages": [HumanMessage(content=question)]}
        result: BaseMessage = self.agent.run(inputs)
        llm = self.chatmodel.with_structured_output(QuestionAnsweredCheck)
        test_response: QuestionAnsweredCheck = llm.invoke(
            f"Question:{question} \n Fact:{fact} \n Answer:{result.content}"
        )
        self.assertTrue(test_response.answered_sufficiently)

    def test_multiple_question_answer_ability(self):
        test_cases = [
            {"question": "Is there a car rental?", "fact": "There is a car rental."},
            {"question": "Which numbers can i call in case of an emergency?",
             "fact": "You can call: 112 International Emergency Call, 122 Fire Service, 133 Police, 140 Alpine-Emergency Call, 141 Doctor-Emergency Service, 144 Ambulance"},
            {"question": "How does the breakfast work?",
             "fact": "The BIO-breakfast basket will be delivered to the door at approximately 8 am the morning after your arrival. "},
            {"question": "How does the breakfast service work? is it for free?",
             "fact": "Your BIO-breakfast basket will be delivered to your door at approximately 8 am the morning after your arrival. Your basket includes cereals, fresh juice, fresh farm eggs, cold meat, fruit of the season, cheese, butter, milk, herbal tea, drinking chocolate & homemade jam. Order your bread and refillss for the following day before 11am online  in my.oha.at. Most of the products are from the local farmer´s market. Please be sure to visit so you can take some of their fresh products home with you. The breakfast is NOT free."},
            {"question": "What are the opening hours of the reception?",
             "fact": "Opening hours: daily 8am to midday and 3pm to 5pm. Call any time between 8am and 9pm Reception"},
            {"question": "Search if dogs are allowed and if yes, what are the costs and multiply them by 6.",
             "fact": "Dogs are allowed. For the final cleaning 35 is charged. 35 times 6 is 210"},
            {"question": "What are the checkin and checkout times?",
             "fact": "Check is from 3 to 5pm and checkout from 8 to 10 am"},
            {"question": "Where can i go bowling?",
             "fact": "One can go bowling in restaurant Schwungradl in Pfarrwerfen"},
            {"question": "Which doctor is reachable on Wednesday?",
             "fact": "Dr.med.univ. Manfred Geringer or  Dr. Christoph Fürthauer or Dr.med.univ. Othmar Frühmann or Dr. Kay Drabeck is reachable on wednesday"},
            # {"question": "Recommend me some things to do", "fact": "There is no swimming pool."},
            # {"question": "I want to go on a date night. Any recommendations?", "fact": "There is no swimming pool."},
            {"question": "Which doctor is also available on Wednesday if Dr. Christoph Fürthauer has closed?",
             "fact": "Dr.med.univ. Manfred Geringer or Dr.med.univ. Othmar Frühmann or Dr. Kay Drabeck is also available on Wendesday"},
            {"question": "What is the phone number of Dr. Manfred Geringer in Werfen?",
             "fact": "Phone number is 0043 (0) 6468 8283 0"},
        ]

        for case in test_cases:
            with self.subTest(case=case):
                self._check_single_question_answer_ability(
                    case['question'], case['fact']
                )

    def test_single_question_answer_ability(self):
        question = "Is there a car rental?"
        fact = "There is a car rental."
        inputs = {
            "messages": [HumanMessage(content=question)]}
        result: BaseMessage = self.agent.run(inputs)
        llm = self.chatmodel.with_structured_output(
            QuestionAnsweredCheck)
        test_response: QuestionAnsweredCheck = llm.invoke(
            f"Question:{question} \n Fact:{fact} \n Answer:{result.content}")

        return test_response.answered_sufficiently


if __name__ == '__main__':
    unittest.main()
