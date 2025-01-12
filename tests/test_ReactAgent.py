import unittest

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_ollama import ChatOllama as LatestChatOllama
from pydantic import BaseModel, Field

from src.ModelTypes.modelTypes import Modeltype
from src.ReasoningLanggraphLLM import ReasoningLanggraphLLM
from tests import logger


# Source:https://python.langchain.com/docs/integrations/tools/ddg

class QuestionAnsweredCheck(BaseModel):
    """Checks if a given question is answered sufficiently"""

    question: str = Field(description="The input question")
    fact: str = Field(description="A fact")
    answer: str = Field(
        description="The answer that should answer the question"
    )
    answered_sufficiently: bool = Field(
        desription="Indicates with True or False, if the answer answers the question, according to the fact",
        default=False)


class TestVectorDatabase(unittest.TestCase):

    def setUp(self):
        self.agent: ReasoningLanggraphLLM = ReasoningLanggraphLLM()
        self.agent.allow_profanity_check = False
        self.agent.allow_hallucination_check = False
        self.chatmodel: ChatOllama = LatestChatOllama(model=Modeltype.LLAMA3_1_8B.value, temperature=0)
        self.translation_model = LatestChatOllama(model=Modeltype.AYA.value, temperature=0)

    def _clear_agent_memory(self):
        self.agent.reset_memory()

    def _check_single_question_answer_ability(self, question, fact):
        inputs = {"messages": [HumanMessage(content=question)]}
        result: BaseMessage = self.agent.run(inputs)
        messages = [
            SystemMessage(
                content=f"""You are a helpful AI assistant assigned to check if a given answer answers a question given the fact. Keep your answer grounded to the input given. If the answer does not answer the question, return only 'FALSE' as the answer. If the answer answers the question return only 'TRUE'."""
            ),
            HumanMessage(
                content=f"""Question:
                    {question}

                    Fact:
                    {fact}

                    Answer:
                    {result.content}"""
            )
        ]

        qa_check_response: BaseMessage = self.chatmodel.invoke(messages)

        response_correct: bool = qa_check_response.content.lower() == "true"

        logger.info(f"Question: {question}")
        logger.info(f"Fact: {fact}")
        logger.info(f"Answer: {result.content}")
        logger.info(f"Verdict: {qa_check_response.content}")
        self.assertTrue(response_correct)

    def test_multiple_question_answer_ability(self):
        test_cases = [
            {"question": "Is there a car rental?", "fact": "There is a car rental."},
            {"question": "Which numbers can i call in case of an emergency?",
             "fact": "You can call: 112 International Emergency Call, 122 Fire Service, 133 Police, 140 Alpine-Emergency Call, 141 Doctor-Emergency Service, 144 Ambulance"},
            {"question": "How does the breakfast work?",
             "fact": "The BIO-breakfast basket will be delivered to the door at approximately 8 am the morning after your arrival. "},
            {"question": "How does the breakfast service work? Is it for free?",  # cannot be answered correctly
             "fact": "Your BIO-breakfast basket will be delivered to your door at approximately 8 am the morning after your arrival. Your basket includes cereals, fresh juice, fresh farm eggs, cold meat, fruit of the season, cheese, butter, milk, herbal tea, drinking chocolate & homemade jam. Order your bread and refillss for the following day before 11am online  in my.oha.at. Most of the products are from the local farmer´s market. Please be sure to visit so you can take some of their fresh products home with you. The breakfast is NOT free."},
            {"question": "What are the opening hours of the reception?",
             "fact": "Opening hours: daily 8am to midday and 3pm to 5pm. Call any time between 8am and 9pm Reception"},
            {"question": "Search if dogs are allowed and if yes, what are the costs and multiply them by 6.",
             "fact": "Dogs are allowed. For the final cleaning 35 is charged. 35 times 6 is 210"},
            {"question": "What are the checkin and checkout times according to the document?",
             "fact": "Check is from 3 to 5pm and checkout from 8 to 10 am"},
            {"question": "Where can i go bowling? Search in the documents",
             "fact": "One can go bowling in restaurant Schwungradl in Pfarrwerfen"},
            {"question": "Which doctor is reachable on Wednesday?",
             "fact": "Dr.med.univ. Manfred Geringer or  Dr. Christoph Fürthauer or Dr.med.univ. Othmar Frühmann or Dr. Kay Drabeck is reachable on wednesday"},
            {"question": "Recommend me some things to do, in my documents",
             "fact": """One could go into a sauna, eat dinner from Restaurant Chili, visit High Rope Climing Garden, go bowling, visit the Farmers Market, go to the gold course, visit a hairdresser, go hiking, go paragliding, get a massage, visit the Moutain Huts, visit Restaurants.
            """},
            {"question": "I want to go on a date night. Any recommendations? Search in the documents",
             "fact": "There is no swimming pool."},
            {"question": "Which doctor is also available on Wednesday if Dr. Christoph Fürthauer has closed?",
             "fact": "Dr.med.univ. Manfred Geringer or Dr.med.univ. Othmar Frühmann or Dr. Kay Drabeck is also available on Wendesday"},
            {"question": "What is the phone number of Dr. Manfred Geringer in Werfen?",
             "fact": "Phone number is 0043 (0) 6468 8283 0"},
        ]

        for case in test_cases:
            with self.subTest(case=case):
                self._clear_agent_memory()
                self._check_single_question_answer_ability(
                    case['question'], case['fact']
                )

    def test_multiple_question_answer_ability_french(self):
        test_cases = [
            {'question': 'Y a-t-il un service de location de voitures ?',
             'english_question': 'Is there a car rental?',
             'fact': 'There is a car rental.'},
            {'question': 'Quels numéros puis-je appeler en cas d urgence ?',
             'english_question': 'Which numbers can i call in case of an emergency?',
             'fact': 'You can call: 112 International Emergency Call, 122 Fire Service, 133 Police, 140 Alpine-Emergency Call, 141 Doctor-Emergency Service, 144 Ambulance'},
            {'question': 'Comment fonctionne le petit-déjeuner ?', 'english_question': 'How does the breakfast work?',
             'fact': 'The BIO-breakfast basket will be delivered to the door at approximately 8 am the morning after your arrival. '},
            {'question': 'Comment fonctionne le service de petit-déjeuner ? Est-ce gratuit ?',
             'english_question': 'How does the breakfast service work? Is it for free?',
             'fact': 'Your BIO-breakfast basket will be delivered to your door at approximately 8 am the morning after your arrival. Your basket includes cereals, fresh juice, fresh farm eggs, cold meat, fruit of the season, cheese, butter, milk, herbal tea, drinking chocolate & homemade jam. Order your bread and refillss for the following day before 11am online  in my.oha.at. Most of the products are from the local farmer´s market. Please be sure to visit so you can take some of their fresh products home with you. The breakfast is NOT free.'},
            {'question': 'Quelles sont les heures d ouverture de la réception ?',
             'english_question': 'What are the opening hours of the reception?',
             'fact': 'Opening hours: daily 8am to midday and 3pm to 5pm. Call any time between 8am and 9pm Reception'},
            {
                'question': 'Rechercher si les chiens sont autorisés et si oui, quels sont les coûts et les multiplier par 6.',
                'english_question': 'Search if dogs are allowed and if yes, what are the costs and multiply them by 6.',
                'fact': 'Dogs are allowed. For the final cleaning 35 is charged. 35 times 6 is 210'},
            {'question': 'Quelles sont les heures d enregistrement et de départ selon le document ?',
             'english_question': 'What are the checkin and checkout times according to the document?',
             'fact': 'Check is from 3 to 5pm and checkout from 8 to 10 am'},
            {'question': 'Où puis-je aller au bowling ? Rechercher dans les documents',
             'english_question': 'Where can i go bowling? Search in the documents',
             'fact': 'One can go bowling in restaurant Schwungradl in Pfarrwerfen'},
            {'question': 'Quel médecin est joignable le mercredi ?',
             'english_question': 'Which doctor is reachable on Wednesday?',
             'fact': 'Dr.med.univ. Manfred Geringer or  Dr. Christoph Fürthauer or Dr.med.univ. Othmar Frühmann or Dr. Kay Drabeck is reachable on wednesday'},
            {'question': 'Recommandez-moi des choses à faire, dans mes documents',
             'english_question': 'Recommend me some things to do, in my documents',
             'fact': 'One could go into a sauna, eat dinner from Restaurant Chili, visit High Rope Climing Garden, go bowling, visit the Farmers Market, go to the gold course, visit a hairdresser, go hiking, go paragliding, get a massage, visit the Moutain Huts, visit Restaurants.'},
            {'question': 'Je veux aller à un rendez-vous amoureux. Des recommandations ? Rechercher dans les documents',
             'english_question': 'I want to go on a date night. Any recommendations? Search in the documents',
             'fact': 'There is no swimming pool.'},
            {'question': 'Quel médecin est également disponible le mercredi si le Dr Christoph Fürthauer est fermé ?',
             'english_question': 'Which doctor is also available on Wednesday if Dr. Christoph Fürthauer has closed?',
             'fact': 'Dr.med.univ. Manfred Geringer or Dr.med.univ. Othmar Frühmann or Dr. Kay Drabeck is also available on Wendesday'},
            {'question': 'Quel est le numéro de téléphone du Dr Manfred Geringer à Werfen ?',
             'english_question': 'What is the phone number of Dr. Manfred Geringer in Werfen?',
             'fact': 'Phone number is 0043 (0) 6468'}
        ]

        for case in test_cases:
            with self.subTest(case=case):
                self._clear_agent_memory()
                self._check_single_question_answer_ability(
                    case['question'], case['fact']
                )

    def test_foreign_question_answer_ability(self):
        test_cases = [
            {"question": "Que s'est-il passé récemment avec Liam Payne?",
             "english_question": "What recently happened with Liam Payne?",
             "fact": "Liam Payne died."},
        ]

        for case in test_cases:
            with self.subTest(case=case):
                self._clear_agent_memory()
                self._check_single_foreign_question_answer_ability(
                    case['question'], case['english_question'], case['fact']
                )

    def _check_single_foreign_question_answer_ability(self, question, english_question, fact):
        inputs = {"messages": [HumanMessage(content=question)]}
        result: BaseMessage = self.agent.run(inputs)

        translation_messages = [
            SystemMessage(
                content="You are a professional translator. Your job is to translate the user input into english. Only respond with the translated sentence."),
            result
        ]
        translated_response: BaseMessage = self.translation_model.invoke(translation_messages)

        messages = [
            SystemMessage(
                content=f"""You are a helpful AI assistant assigned to check if a given answer answers a question given the fact. Keep your answer grounded to the input given. If the answer does not answer the question, return only 'FALSE' as the answer. If the answer answers the question return only 'TRUE'."""
            ),
            HumanMessage(
                content=f"""Question:
                    {english_question}

                    Fact:
                    {fact}

                    Answer:
                    {translated_response.content}"""
            )
        ]

        qa_check_response: BaseMessage = self.chatmodel.invoke(messages)

        logger.debug(f"Question: {question}")
        logger.debug(f"Fact: {fact}")
        logger.debug(f"Answer: {result.content}")
        logger.debug(f"Verdict: {qa_check_response.content}")

        self.assertTrue(qa_check_response.content.lower() == "true")

    @unittest.skip("Disabled because no longer needed for regular testing")
    def test_testcase_repeatedly(self):

        test_case = {"question": "What are the opening hours of the reception?",
                     "fact": "Opening hours: daily 8am to midday and 3pm to 5pm. Call any time between 8am and 9pm Reception"}

        num_tries = 5

        for i in range(num_tries):
            with self.subTest(case=test_case):
                self._clear_agent_memory()
                self._check_single_question_answer_ability(
                    test_case['question'], test_case['fact']
                )

    def test_single_question_answer_ability(self):
        testcase = {"question": "Which doctor is also available on Wednesday if Dr. Christoph Fürthauer has closed?",
                    "fact": "Dr.med.univ. Manfred Geringer or Dr.med.univ. Othmar Frühmann or Dr. Kay Drabeck is also available on Wendesday"}
        question = testcase['question']
        fact = testcase['fact']

        self._check_single_question_answer_ability(question, fact)


if __name__ == '__main__':
    unittest.main()
