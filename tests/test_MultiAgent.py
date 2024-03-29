import unittest

from langchain_community.tools import DuckDuckGoSearchResults


# Source:https://python.langchain.com/docs/integrations/tools/ddg

class TestVectorDatabase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_querying_the_internet_works(self):
        search = DuckDuckGoSearchResults()
        web_result = search.run("What is langchain")
        expected_result = "LangChain is a powerful tool that can be used to build a wide range of LLM-powered applications"
        self.assertTrue(expected_result in web_result)


if __name__ == '__main__':
    unittest.main()
