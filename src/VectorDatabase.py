import io
import json
import logging
import os
from pathlib import Path, WindowsPath
from typing import List, Tuple, Dict, Set

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents.base import Document
from langchain_core.tools import tool
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader
import nltk

from DocumentSummariser import DocumentSummariser
from DocumentWrapper import DocumentWrapper
from FileAddedObservable import FileAddedObservable

logger = logging.getLogger(__name__)


class DocumentVectorStorage(FileAddedObservable):

    def __init__(self):
        super().__init__()
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')

        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self.INDEXED_FILES_PATH = self.PROJECT_ROOT / 'indexedFiles'
        self.DATABASE_PATH: WindowsPath = str(self.PROJECT_ROOT / 'chroma_db')
        # self.EMBEDDING_CACHE: WindowsPath = str(self.PROJECT_ROOT / 'embedding-models/all-miniLM')
        self.EMBEDDING_CACHE: WindowsPath = str(self.PROJECT_ROOT / 'embedding-models/bge')
        self.DocumentSummariser = DocumentSummariser()

        model_name = "BAAI/bge-small-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        query_instruction = "Represent this sentence for searching relevant passages:"
        encode_kwargs = {'normalize_embeddings': True, "default_prompt_name": query_instruction}
        embed_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            # query_instruction=query_instruction,
            cache_folder=str(self.EMBEDDING_CACHE)
        )

        """embed_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                                                   cache_folder=str(self.EMBEDDING_CACHE),
                                                                   model_kwargs={'device': 'cpu'})"""

        self.semantic_chunker: SemanticChunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile",
                                                                 breakpoint_threshold_amount=80.0)

        self.db: Chroma = Chroma(persist_directory=str(self.DATABASE_PATH), embedding_function=embed_model)

        indexed_filenames: List[str] = list(set([item["source"] for item in self.db.get()["metadatas"]]))

        unindexed_files = self.extract_unindexed_files(indexed_filenames)
        self.index_new_files(unindexed_files)

    def index_new_files(self, unindexed_filenames: List[str]) -> None:
        for each_filename in unindexed_filenames:
            try:
                filetype: str = os.path.splitext(each_filename)[1].lower()
                doc: List[Document] = self.load_file(each_filename, filetype)
                splitted_docs: List[Document] = self.semantic_chunker.split_documents(doc)

                self.add_documents_chunked(splitted_docs)

            except Exception as e:
                print(e)

    def add_documents_chunked(self, splitted_docs):
        chunk_size = 10
        for i in range(0, len(splitted_docs), chunk_size):
            chunk = splitted_docs[i:i + chunk_size]
            self.db.add_documents(chunk)

    def extract_unindexed_files(self, indexed_filenames: List[str]) -> List[str]:
        unindexed_files: List[str] = []
        for root, dirs, filenames in os.walk(self.INDEXED_FILES_PATH):
            for each_filename in filenames:
                if each_filename not in indexed_filenames:
                    unindexed_files.append(each_filename)
        return unindexed_files

    def load_file(self, filename: str, filetype: str) -> List[Document]:
        doc: List[Document] = []
        file_path: str = os.path.join(os.getcwd(), self.INDEXED_FILES_PATH, filename)

        if 'json' in filetype:
            with open(file_path) as f:
                json_str: str = json.dumps(json.load(f))
            doc = [Document(page_content=json_str, metadata={"source": filename})]
        elif 'pdf' in filetype:
            loader = PyPDFLoader(file_path=file_path)
            doc = loader.load()
            doc[0].metadata["source"] = filename
        else:
            doc: List[Document]
            try:
                loader = UnstructuredLoader(file_path)
                loaded_doc = loader.load()

                whole_content = "\n".join(text.page_content for text in loaded_doc)
                doc_summary = self.DocumentSummariser.summarize_text(whole_content)

                merged_doc: Document = Document(page_content=whole_content,
                                                metadata={"source": filename, "docsummary": doc_summary})
                doc.append(merged_doc)
                self.notify_file_added()

            except Exception as e:
                print(e)
        return doc

    def query_vector_database(self, query: str) -> List[Document]:
        return self.db.similarity_search(query)

    def query_vector_database_with_keywords(self, queries: List[str]):
        docs = set()
        for each_query in queries:
            results = self.db.search(query=each_query, search_type="mmr", k=15)
            for each_doc in results:
                converted_doc = DocumentWrapper(each_doc)
                docs.add(converted_doc)
        return docs

    def query_vector_database_with_relevance(self, query: str) -> List[Tuple[Document, float]]:
        return self.db.similarity_search_with_relevance_scores(query)

    def index_new_file(self, filepath: str) -> None:
        loader = PyPDFLoader(filepath)
        docs: list[Document] = loader.load()
        splitted_docs: list[Document] = self.semantic_chunker.split_documents(docs)
        self.add_documents_chunked(splitted_docs)

    def get_indexed_filenames(self) -> List[str]:
        """
        Get the list of indexed filenames from the database metadata.

        Returns:
            List[str]: A list of indexed filenames.
        """
        if self.db:
            metadata: List[Dict] = self.db.get()["metadatas"]
            indexed_filenames: List[str] = list(set([item["source"] for item in metadata]))
            return indexed_filenames
        else:
            return []

    def get_all_chunks(self) -> List[str]:
        """
        Get all chunks from the vector database.

        Returns:
            List[str]: A list of all chunks.
        """
        all_chunks: List[Document] = self.db.get()["documents"]
        all_filenames: List[str] = [item["source"] for item in self.db.get()["metadatas"]]
        summaries = [item.get("docsummary", "No summary") for item in self.db.get()["metadatas"]]

        return all_chunks, all_filenames, summaries

    def get_document_summaries(self) -> List[str]:
        all_summaries: List[str] = [item["docsummary"] for item in self.db.get()["metadatas"] if "docsummary" in item]
        return list(set(all_summaries))  # unique summaries

    def process_and_index_file(self, uploaded_file: io.BytesIO) -> None:
        """
        Processes an uploaded file and indexes it into the Chroma database.

        Args:
            uploaded_file (io.BytesIO): The file uploaded by the user.

        Returns:
            None
        """

        byte: bytes = io.BytesIO(uploaded_file.getvalue()).read()
        filename: str = uploaded_file.name

        current_path: str = os.getcwd()

        filepath: str = os.path.join(current_path, "indexedFiles", filename)

        with open(f"{filepath}", "wb") as f:
            f.write(byte)

        filetype: str = uploaded_file.type

        doc: list[Document] = self.load_file(filename, filetype)

        splitted_docs: list[Document] = self.semantic_chunker.split_documents(doc)

        try:
            self.add_documents_chunked(splitted_docs)
        except Exception as e:
            print(e)

    @tool
    def get_local_knowledge(self, query: str) -> List[Document]:
        """
        Get potential knowledge about the query on the current computer
        :param query: The question
        :return: A list of entries of additional Information. Can be empty.
        """
        return self.db.as_retriever(query)

    def remove_all_documents(self):

        self.db.reset_collection()

        indexed_files_path = os.path.join(os.getcwd(), self.INDEXED_FILES_PATH)

        try:
            for filename in os.listdir(indexed_files_path):
                file_path = os.path.join(indexed_files_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"Error: {e}")
