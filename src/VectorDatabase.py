import logging
import os
from typing import List

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter

from src.configuration.logger_config import setup_logging

logger = setup_logging()


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


def find_directory(start_path, target_name) -> str:
    """
    Search recursively for a directory with the specified name starting from start_path.

    :param start_path: str - Path where the search starts.
    :param target_name: str - Name of the directory to search for.
    :return: str - Full path to the found directory or original path if not found.
    """
    for root, dirs, _ in os.walk(start_path):
        if target_name in dirs:
            return os.path.join(root, target_name)
    return ""


class DocumentVectorStorage:
    def __init__(self):
        base_search_path = "/"
        self.DATABASE_PATH: str = find_directory(base_search_path, "chroma_db")
        self.INDEXED_FILES_PATH = find_directory(base_search_path, "indexedFiles")

        self.db = None

        if directory_has_contents(self.DATABASE_PATH):
            self.db = Chroma(persist_directory=self.DATABASE_PATH,
                             embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

            indexed_filenames: List[str] = list(set([item["source"] for item in self.db.get()["metadatas"]]))

            if not all_files_indexed(self.INDEXED_FILES_PATH, indexed_filenames):
                logging.warning("Not all files are indexed. Be Cautious!")

        else:
            loader = DirectoryLoader(self.INDEXED_FILES_PATH,
                                     use_multithreading=True)  # https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
            docs: list = loader.load()
            # json_docs: list = DirectoryLoader(self.INDEXED_FILES_PATH, glob="**/*.json", loader_cls=TextLoader).load()
            # docs.append(json_docs)
            # Json docs können nicht mit split_documents gesplitted werden. Brauchen RecursiveJsonSplitter. Quelle:  https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_json_splitter
            # Liste an Document Loadern: https://stackoverflow.com/questions/77057531/loading-different-document-types-in-langchain-for-an-all-data-source-qa-bot

            logging.debug(f"{len(docs)} documents were imported")
            splitted_docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)

            self.db = Chroma.from_documents(splitted_docs, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
                                            persist_directory=self.DATABASE_PATH)

    def query_vector_database(self, query: str):
        return self.db.similarity_search(query)

    def index_new_file(self, filepath: str):
        loader = PyPDFLoader(filepath)
        docs: list = loader.load()
        splitted_docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)
        self.db.add_documents(splitted_docs)

    def get_indexed_filenames(self) -> List[str]:
        """
        Get the list of indexed filenames from the database metadata.

        Returns:
            List[str]: A list of indexed filenames.
        """
        if self.db:
            metadata = self.db.get()["metadatas"]
            indexed_filenames = list(set([item["source"] for item in metadata]))
            return indexed_filenames
        else:
            return []

    def get_all_chunks(self) -> List[str]:
        """
        Get all chunks from the vector database.

        Returns:
            List[str]: A list of all chunks.
        """
        all_chunks = []

        if self.db:
            documents = self.db.get()["documents"]
            for doc in documents:
                all_chunks.extend(documents)

        return all_chunks

    def process_and_index_file(self, uploaded_file):
        """
        Processes an uploaded file and indexes it into the Chroma database.

        Args:
            uploaded_file (io.BytesIO): The file uploaded by the user.

        Returns:
            None
        """

        import io

        byte = io.BytesIO(uploaded_file.getvalue()).read()
        filename = uploaded_file.name
        with open(f"../indexedFiles/{filename}", "wb") as f:
            f.write(byte)

        # Create a FileLoader instance
        from langchain_community.document_loaders import UnstructuredFileLoader

        filetype: str = uploaded_file.type

        from langchain_core.documents.base import Document

        doc = None
        if 'json' in filetype:
            import json
            json_str = None
            with open(f"../indexedFiles/{filename}") as f:
                json_str = json.dumps(json.load(f))
            doc = [Document(page_content=json_str,
                            metadata={"source": f"../indexedFiles/{filename}"})]

        else:
            loader = UnstructuredFileLoader(f"../indexedFiles/{filename}")
            doc = loader.load()

        splitted_docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(doc)

        self.db.add_documents(splitted_docs)

        """from langchain_core.documents.base import Document

        # documents = [uploaded_file.read().decode('utf-8')]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        doc = Document(page_content=uploaded_file.read().decode('utf-8'),
                       metadata={"source": f"manual_upload/{uploaded_file.name}"})

        splitted_docs = text_splitter.split_documents([doc])  # [] to convert to list
        # splitted_docs = text_splitter.split_text(documents[0])

        # DirectoryLoader(self.INDEXED_FILES_PATH,use_multithreading=True).load()[0].metadata

        self.db.add_documents(splitted_docs)"""

    @tool
    def get_local_knowledge(self, query: str):
        """
        Get potential knowledge about the query on the current computer
        :param query: The question
        :return: A list of entries of additional Information. Can be empty.
        """
        return self.db.as_retriever(query)


if '__main__':
    vectordb = DocumentVectorStorage()
