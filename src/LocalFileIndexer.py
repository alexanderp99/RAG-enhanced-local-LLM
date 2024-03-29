import logging
import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

logging.basicConfig(level=logging.WARN)


def directory_has_contents(dir_path):
    """
    Checks if a given directory exists and has subfiles or subdirectories within.

    Parameters:
    dir_path (str): The path to the directory to check.

    Returns:
    bool: True if the directory exists and has contents, False otherwise.
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        contents = os.listdir(dir_path)
        if contents:
            return True
        else:
            return False
    else:
        # Directory does not exist
        return False


def extract_filenames_from_directory(dir_path: str) -> List[str]:
    """
    Extracts all file names from the given directory.

    Parameters:
    dir_path (str): The path to the directory from which to extract file names.

    Returns:
    list: A list of file names (as strings) found in the given directory.
    """
    file_names = []
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        for item in os.listdir(dir_path):
            full_path = os.path.join(dir_path, item)
            if os.path.isfile(full_path):
                file_names.append(item)
    return file_names


def all_files_indexed(directory_path: str, indexed_filenames: List[str]) -> bool:
    """
    Checks if all files from the directory are indexed in the database.

    Parameters:
    directory_path (str): The path to the directory containing files.
    indexed_filenames (List[str]): List of filenames indexed in the database.

    Returns:
    bool: True if all files from the directory are indexed, False otherwise.
    """
    indexed_filenames_only = [os.path.basename(path) for path in indexed_filenames]

    for filename in extract_filenames_from_directory(directory_path):
        if filename not in indexed_filenames_only:
            return False
    return True


class DocumentVectorStorage:
    def __init__(self):
        self.DATABASE_PATH: str = "../chroma_db"
        self.INDEXED_FILES_PATH: str = "../indexedFiles"
        self.db = None

        if directory_has_contents(self.DATABASE_PATH):
            self.db = Chroma(persist_directory=self.DATABASE_PATH,
                             embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

            indexed_filenames: List[str] = list(set([item["source"] for item in self.db.get()["metadatas"]]))

            if not all_files_indexed(self.INDEXED_FILES_PATH, indexed_filenames):
                raise Exception("Not all files are indexed. Be Cautious!")

        else:
            loader = DirectoryLoader(self.INDEXED_FILES_PATH,
                                     use_multithreading=True)  # https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory

            docs = loader.load()
            logging.debug(f"{len(docs)} documents were imported")

            splitted_docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)

            self.db = Chroma.from_documents(splitted_docs, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
                                            persist_directory=self.DATABASE_PATH)
            self.db.persist()

    def query_vector_database(self, query: str):
        return self.db.similarity_search(query)

    def index_new_file(self, file: UploadedFile):
        splitted_docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(file)
        self.db._collection.add(splitted_docs)

    @tool
    def get_local_knowledge(self, query: str):
        """
        Get potential knowledge about the query on the current computer
        :param query: The question
        :return: A list of entries of additional Information. Can be empty.
        """
        return self.db.as_retriever(query)
