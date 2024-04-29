import os
from typing import List

import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter


def directory_has_contents(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        contents = os.listdir(dir_path)
        return bool(contents)
    return False


def extract_filenames_from_directory(dir_path: str) -> List[str]:
    file_names = []
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        for item in os.listdir(dir_path):
            full_path = os.path.join(dir_path, item)
            if os.path.isfile(full_path):
                file_names.append(item)
    return file_names


def all_files_indexed(directory_path: str, indexed_filenames: List[str]) -> bool:
    indexed_filenames_only = [os.path.basename(path) for path in indexed_filenames]
    for filename in extract_filenames_from_directory(directory_path):
        if filename not in indexed_filenames_only:
            return False
    return True


def find_directory(start_path, target_name):
    for root, dirs, _ in os.walk(start_path):
        if target_name in dirs:
            return os.path.join(root, target_name)
    return None


class DocumentVectorStorage:
    def __init__(self):
        base_search_path = "/"
        self.DATABASE_PATH = find_directory(base_search_path, "chroma_db")
        self.INDEXED_FILES_PATH = find_directory(base_search_path, "indexedFiles")

        self.client = None
        self.collection = None

        if directory_has_contents(self.DATABASE_PATH):

            self.client = chromadb.PersistentClient(path=self.DATABASE_PATH)

            self.collection = self.client.create_collection(name="your-stuff")


        else:
            loader = DirectoryLoader(self.INDEXED_FILES_PATH, use_multithreading=True)
            docs = loader.load()

            splitted_docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)

            self.collection.add(splitted_docs)

    def query_vector_database(self, query: str):
        return self.db.similarity_search(query)

    def index_new_file(self, filepath: str):
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        splitted_docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)
        self.db.add_documents(splitted_docs)
        self.db.persist()

    def get_indexed_filenames(self) -> List[str]:
        if self.db:
            metadata = self.db.get()["metadatas"]
            return list(set([item["source"] for item in metadata]))
        return []

    def get_all_chunks(self) -> List[str]:
        all_chunks = []
        if self.db:
            documents = self.db.get()["documents"]
            for doc in documents:
                all_chunks.extend(documents)
        return all_chunks

    def process_and_index_file(self, uploaded_file):
        documents = [uploaded_file.read().decode('utf-8')]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        splitted_docs = text_splitter.split_documents(documents)
        self.db.add_documents(splitted_docs)
        self.db.persist()

    @tool
    def get_local_knowledge(self, query: str):
        return self.db.as_retriever(query)


if '__main__':
    pass
    vectordb = DocumentVectorStorage()
