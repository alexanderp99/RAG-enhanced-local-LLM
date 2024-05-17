import io
import os
from typing import List

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents.base import Document
from langchain_core.tools import tool
from langchain_experimental.text_splitter import SemanticChunker

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
        self.DATABASE_PATH: str = "./chroma_db"
        self.INDEXED_FILES_PATH = "./indexedFiles"
        """self.EMBEDDING_FUNCTION = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2")"""  # Maybe also parallellize loading data: https://www.youtube.com/watch?v=7FvdwwvqrD4
        self.CHUNK_OVERLAP = 0
        self.CHUNK_SIZE = 1000
        # embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        # embed_model = load_model("all-MiniLM-L6-v2")
        """embed_model = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2")"""
        embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                            cache_folder="./embedding-models/all-miniLM",
                                            model_kwargs={'device': 'cpu'})

        self.semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")

        self.db = Chroma(persist_directory=self.DATABASE_PATH, embedding_function=embed_model)

        indexed_filenames: List[str] = list(set([item["source"] for item in self.db.get()["metadatas"]]))

        unindexed_files = self.extract_unindexed_files(indexed_filenames)
        self.index_new_files(unindexed_files)

        """else:
            loader = DirectoryLoader(self.INDEXED_FILES_PATH,
                                     use_multithreading=True)  # https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
            docs: list = loader.load()
            # json_docs: list = DirectoryLoader(self.INDEXED_FILES_PATH, glob="**/*.json", loader_cls=TextLoader).load()
            # docs.append(json_docs)
            # Json docs können nicht mit split_documents gesplitted werden. Brauchen RecursiveJsonSplitter. Quelle:  https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_json_splitter
            # Liste an Document Loadern: https://stackoverflow.com/questions/77057531/loading-different-document-types-in-langchain-for-an-all-data-source-qa-bot

            logging.debug(f"{len(docs)} documents were imported")
            splitted_docs = CharacterTextSplitter(chunk_size=self.CHUNK_SIZE,
                                                  chunk_overlap=self.CHUNK_OVERLAP).split_documents(docs)

            self.db = Chroma.from_documents(splitted_docs, self.EMBEDDING_FUNCTION,
                                            persist_directory=self.DATABASE_PATH)"""

    def download_and_load_model(self, model_name):
        """
        used to Download a model. Could be used as script before running the streamlit application.
        :param model_name:
        :return:
        """
        from sentence_transformers import models
        model_path = os.path.join(os.getcwd(), model_name)
        if not os.path.exists(model_path):
            # Download the model if it doesn't exist
            os.makedirs(model_path, exist_ok=True)
            model = models.Transformer(model_name)
            model.save(model_path)
        else:
            model = models.Transformer(model_path)

        # Load SentenceTransformer with downloaded model
        embedding_function = SentenceTransformerEmbeddings(model)
        return embedding_function

    def index_new_files(self, unindexed_filenames: List[str]):

        for each_filename in unindexed_filenames:
            filetype = os.path.splitext(each_filename)[1].lower()

            doc = self.load_file(each_filename, filetype)

            splitted_docs = self.semantic_chunker.split_documents(doc)

            """splitted_docs = CharacterTextSplitter(chunk_size=self.CHUNK_SIZE,
                                                  chunk_overlap=self.CHUNK_OVERLAP).split_documents(doc)"""
            self.db.add_documents(splitted_docs)

    def extract_unindexed_files(self, indexed_filenames):
        unindexed_files = []
        for root, dirs, filenames in os.walk(self.INDEXED_FILES_PATH):
            for each_filename in filenames:
                if each_filename not in indexed_filenames:
                    unindexed_files.append(each_filename)
        return unindexed_files

    def load_file(self, filename, filetype):
        doc = None
        if '.json' in filetype:
            import json
            json_str = None
            with open(filename) as f:
                json_str = json.dumps(json.load(f))
            doc = [Document(page_content=json_str, metadata={"source": filename})]
        else:
            file_path = os.path.join(os.getcwd(), self.INDEXED_FILES_PATH, filename)
            loader = UnstructuredFileLoader(file_path)
            print(f"current path:")
            doc = loader.load(encoding="utf-8")
            doc[0].metadata["source"] = filename
        return doc

    def query_vector_database(self, query: str):
        return self.db.similarity_search(query)

    def query_vector_database_with_relevance(self, query: str):
        return self.db.similarity_search_with_relevance_scores(query)

    def index_new_file(self, filepath: str):
        loader = PyPDFLoader(filepath)
        docs: list = loader.load()
        """splitted_docs = CharacterTextSplitter(chunk_size=self.CHUNK_SIZE,
                                              chunk_overlap=self.CHUNK_OVERLAP).split_documents(docs)"""
        splitted_docs = self.semantic_chunker.split_documents(docs)
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
        all_chunks = self.db.get()["documents"]
        all_filenames = [item["source"] for item in self.db.get()["metadatas"]]

        return all_chunks, all_filenames

    def process_and_index_file(self, uploaded_file):
        """
        Processes an uploaded file and indexes it into the Chroma database.

        Args:
            uploaded_file (io.BytesIO): The file uploaded by the user.

        Returns:
            None
        """

        byte = io.BytesIO(uploaded_file.getvalue()).read()
        filename = uploaded_file.name

        current_path: str = os.getcwd()

        filepath = os.path.join(current_path, "indexedFiles", filename)
        print("src not in filepath")

        with open(f"{filepath}", "wb") as f:
            f.write(byte)

        filetype: str = uploaded_file.type

        doc = self.load_file(filename, filetype)

        """splitted_docs = CharacterTextSplitter(chunk_size=self.CHUNK_SIZE,
                                              chunk_overlap=self.CHUNK_OVERLAP).split_documents(doc)"""
        splitted_docs = self.semantic_chunker.split_documents(doc)

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
