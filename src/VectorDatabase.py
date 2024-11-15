import io
import json
import logging
import os
# setup_logging()
# logger = get_logger(__name__)
# logger = logging.getLogger(__name__)
from pathlib import Path, WindowsPath
from typing import List, Tuple, Dict

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents.base import Document
from langchain_core.tools import tool
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from .test import Dummy

logger = logging.getLogger(__name__)


class DocumentVectorStorage:

    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self.INDEXED_FILES_PATH = self.PROJECT_ROOT / 'indexedFiles'
        self.DATABASE_PATH: WindowsPath = str(self.PROJECT_ROOT / 'chroma_db')
        self.EMBEDDING_CACHE: WindowsPath = str(self.PROJECT_ROOT / 'embedding-models/all-miniLM')

        embed_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                                                   cache_folder=str(self.EMBEDDING_CACHE),
                                                                   model_kwargs={'device': 'cpu'})

        self.semantic_chunker: SemanticChunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")

        self.db: Chroma = Chroma(persist_directory=str(self.DATABASE_PATH), embedding_function=embed_model)

        indexed_filenames: List[str] = list(set([item["source"] for item in self.db.get()["metadatas"]]))

        unindexed_files = self.extract_unindexed_files(indexed_filenames)
        self.index_new_files(unindexed_files)

        logger.debug("VectorDB CTOR 1!")
        logger.debug("VectorDB CTOR 2!")

        Dummy.run_script()

        logger.debug("VectorDB CTOR 3!")

    def index_new_files(self, unindexed_filenames: List[str]) -> None:
        for each_filename in unindexed_filenames:
            filetype: str = os.path.splitext(each_filename)[1].lower()
            doc: List[Document] = self.load_file(each_filename, filetype)
            splitted_docs: List[Document] = self.semantic_chunker.split_documents(doc)
            self.db.add_documents(splitted_docs)

    def extract_unindexed_files(self, indexed_filenames: List[str]) -> List[str]:
        unindexed_files: List[str] = []
        for root, dirs, filenames in os.walk(self.INDEXED_FILES_PATH):
            for each_filename in filenames:
                if each_filename not in indexed_filenames:
                    unindexed_files.append(each_filename)
        return unindexed_files

    def load_file(self, filename: str, filetype: str) -> List[Document]:
        doc: List[Document] = []
        if '.json' in filetype:
            with open(filename) as f:
                json_str: str = json.dumps(json.load(f))
            doc = [Document(page_content=json_str, metadata={"source": filename})]
        else:
            file_path: str = os.path.join(os.getcwd(), self.INDEXED_FILES_PATH, filename)
            loader = UnstructuredFileLoader(file_path)
            doc = loader.load()
            doc[0].metadata["source"] = filename
        return doc

    def query_vector_database(self, query: str) -> List[Document]:
        return self.db.similarity_search(query)

    def query_vector_database_with_relevance(self, query: str) -> List[Tuple[Document, float]]:
        return self.db.similarity_search_with_relevance_scores(query)

    def index_new_file(self, filepath: str) -> None:
        loader = PyPDFLoader(filepath)
        docs: list[Document] = loader.load()
        splitted_docs: list[Document] = self.semantic_chunker.split_documents(docs)
        self.db.add_documents(splitted_docs)

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

        return all_chunks, all_filenames

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

        self.db.add_documents(splitted_docs)

    @tool
    def get_local_knowledge(self, query: str) -> List[Document]:
        """
        Get potential knowledge about the query on the current computer
        :param query: The question
        :return: A list of entries of additional Information. Can be empty.
        """
        return self.db.as_retriever(query)
