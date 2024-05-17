import unittest

from src.VectorDatabase import DocumentVectorStorage


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

    def test_using_simple_retriever_works(self):
        query = "Causal Inference has also been used for "
        result = self.document_vector_storage.db.similarity_search(query)
        self.assertTrue(len(result) > 0)

    def test_unrelevant_question_works(self):
        query = "I love walking on the beach"
        result = self.document_vector_storage.db.similarity_search(query)
        self.assertTrue(len(result) == 0)

    def test_index_new_file(self):
        file_path = "./test_for_indexing_documents/French Revolution - Wikipedia.pdf"

        self.document_vector_storage.index_new_file(file_path)
        query = "When was the storming of the bastille?"
        search_result = self.document_vector_storage.db.similarity_search(query)
        self.assertTrue(len(search_result) > 0)


if __name__ == '__main__':
    unittest.main()
