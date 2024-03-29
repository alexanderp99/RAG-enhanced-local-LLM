import unittest

from src.LocalFileIndexer import DocumentVectorStorage


class TestVectorDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.document_vector_storage = DocumentVectorStorage()

    def test_causal_inference_document_is_indexed(self):
        query = "Causal Inference has also been used for "
        result = self.document_vector_storage.db.similarity_search(query)
        self.assertTrue(len(result) > 0)

    def test_statistics_wikipedia_is_indexed(self):
        query = "Karl Pearson"
        result = self.document_vector_storage.db.similarity_search(query)
        self.assertTrue(len(result) > 0)

    def using_simple_retriever_works(self):
        query = "Causal Inference has also been used for "
        result = self.document_vector_storage.get_local_knowledge(query)
        self.assertTrue(len(result) > 0)


if __name__ == '__main__':
    unittest.main()
