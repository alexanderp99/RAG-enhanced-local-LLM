import io
import json
import logging
import os
from pathlib import Path, WindowsPath
from typing import List, Tuple, Dict

import torch
from pymilvus import connections, FieldSchema, DataType, CollectionSchema, Collection, utility, MilvusClient, RRFRanker, \
    AnnSearchRequest
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_core.tools import tool
from langchain_experimental.text_splitter import SemanticChunker  # For potential future use
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader
import nltk
from langchain_text_splitters import CharacterTextSplitter
from pymilvus import connections, db
from transformers import AutoTokenizer, AutoModel
import pymilvus
import pandas as pd
from pymilvus import MilvusClient
from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections,
)
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
from milvus import default_server
from pymilvus import connections, utility
from pymilvus import connections

logger = logging.getLogger(__name__)


class DocumentVectorStorage:

    def __init__(self):
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')

        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self.INDEXED_FILES_PATH = self.PROJECT_ROOT / 'indexedFiles'
        self.DATABASE_PATH: WindowsPath = str(self.PROJECT_ROOT / 'chroma_db')
        self.DATABASE_PATH: WindowsPath = str(self.PROJECT_ROOT / 'milvus')
        self.TOKENIZER_PATH: WindowsPath = str(self.PROJECT_ROOT / 'tokenizers')
        self.AUTOMODEL_PATH: WindowsPath = str(self.PROJECT_ROOT / 'automodel')

        self.EMBEDDING_CACHE: WindowsPath = str(self.PROJECT_ROOT / 'embedding-models/all-miniLM')

        MODEL = (
            "BAAI/bge-small-en-v1.5"  # Name of model from HuggingFace Models
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=self.TOKENIZER_PATH)
        self.model = AutoModel.from_pretrained(MODEL, cache_dir=self.AUTOMODEL_PATH)

        self.COLLECTION_NAME = "huggingface_test"  # Collection name
        self.DIMENSION = 384  # Embedding dimension depending on model

        self.analyzer = build_default_analyzer(language="en")
        self.MAX_CHUNK_LENGTH = 500

        default_server.start()
        connections.connect(host='127.0.0.1', port=default_server.listen_port)
        print(utility.get_server_version())

        default_server.cleanup()
        self.milvus_client = MilvusClient("./milvus_demo.db")
        self.bm25_ef = BM25EmbeddingFunction(self.analyzer)
        # bm25_ef.fit(documents)

        conn = connections.connect(
            alias="default",
            user='username',
            password='password',
            host='localhost',
            port=default_server.listen_port
        )

        database = db.using_database("default")

        schema = self.miluv_client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=self.MAX_CHUNK_LENGTH,
                         enable_analyzer=True,  # Whether to enable text analysis for this field
                         enable_match=True  # Whether to enable text match
                         )
        # schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=self.DIMENSION)

        index_params = self.miluv_client.prepare_index_params()

        # Add indexes
        index_params.add_index(
            field_name="dense",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )

        """index_params.add_index(
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
            metric_type="IP",  # Currently, only IP (Inner Product) is supported for sparse vectors
        )"""

        if self.miluv_client.has_collection(collection_name=self.Collection_NAME):
            self.miluv_client.drop_collection(collection_name=self.Collection_NAME)
        self.miluv_client.create_collection(
            collection_name=self.Collection_NAME,
            schema=schema,
            index_params=index_params
        )

        """self.semantic_chunker: SemanticChunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile",
                                                                 breakpoint_threshold_amount=95.0)"""
        self.chunker = text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

        # self.db: Chroma = Chroma(persist_directory=str(self.DATABASE_PATH), embedding_function=embed_model)

        indexed_filenames: List[str] = list(set([item["source"] for item in self.db.get()["metadatas"]]))

        unindexed_files = self.extract_unindexed_files(indexed_filenames)
        self.index_new_files(unindexed_files)

    def embed_query(self, query: str):
        return self._embed_dense(query)

    def _embed_dense(self, query):
        queries = [query]
        instruction = "Represent this sentence for searching relevant passages:"
        encoded_input = self.tokenizer([instruction + q for q in queries], padding=True, truncation=True,
                                       return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]

        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def _embed_sparse(self, query):
        product_title_embeddings = self.bm25_ef.encode_documents(query)
        return product_title_embeddings

    def index_new_files(self, unindexed_filenames: List[str]) -> None:
        for each_filename in unindexed_filenames:
            try:
                filetype: str = os.path.splitext(each_filename)[1].lower()
                doc: List[Document] = self.load_file(each_filename, filetype)
                # splitted_docs: List[Document] = self.semantic_chunker.split_documents(doc)

                self.add_documents_chunked(doc)

            except Exception as e:
                print(e)

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
        elif 'txt' in filetype:
            with open(file_path, encoding="utf8") as f:
                file = f.read()

                created_docs = self.chunker.create_documents([file])

                for each_doc in created_docs:
                    each_doc.metadata["source"] = filename
                doc = created_docs

        else:
            doc: List[Document]
            try:
                loader = UnstructuredLoader(file_path)
                loaded_doc = loader.load()

                whole_content = "\n".join(text.page_content for text in loaded_doc)

                merged_doc: Document = Document(page_content=whole_content, metadata={"source": filename})
                doc.append(merged_doc)

            except Exception as e:
                print(e)
        return doc

    """def query_vector_database(self, query: str) -> List[Document]:
        return self.db.similarity_search(query)"""

    def query_vector_database(self, query: str) -> List[Document]:
        # Generate embedding for the query
        query_embedding = self.embed_model.embed_query(query)

        filter = f"TEXT_MATCH(text,'{query}')"

        # Search in Milvus
        search_params = {"params": {"nprobe": 16}}  # Adjust nprobe as needed
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embeddings",
            filter=filter,
            param=search_params,
            limit=10,  # Adjust limit as needed
            output_fields=["text"]
        )

        # Extract and return documents
        return [Document(page_content=result.entity.get("text")) for result in results[0]]

    """def query_vector_database_with_relevance(self, query: str) -> List[Tuple[Document, float]]:
            return self.db.similarity_search_with_relevance_scores(query)"""

    def query_vector_database_with_relevance(self, query: str) -> List[Tuple[Document, float]]:
        # Generate embedding for the query
        query_embedding = self.embed_model.embed_query(query)

        search_param_1 = {
            "data": [query_embedding],
            "anns_field": "dense",
            "param": {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            },
            "limit": 3
        }
        request_1 = AnnSearchRequest(**search_param_1)

        """query_sparse_vector = {3573: 0.34701499565746674}, {5263: 0.2639375518635271}
        search_param_2 = {
            "data": [query_sparse_vector],
            "anns_field": "sparse",
            "param": {
                "metric_type": "IP",
                "params": {}
            },
            "limit": 3
        }
        request_2 = AnnSearchRequest(**search_param_2)"""

        reqs = [request_1]

        ranker = RRFRanker()

        res = self.milvus_client.hybrid_search(
            collection_name=self.COLLECTION_NAME,
            reqs=reqs,
            ranker=ranker,
            limit=5
        )

        res

        # Extract and return documents with scores
        return [(Document(page_content=result.entity.get("text")), result.distance) for result in results[0]]

    """def add_documents_chunked(self, splitted_docs):
            chunk_size = 10
            for i in range(0, len(splitted_docs), chunk_size):
                chunk = splitted_docs[i:i + chunk_size]
                self.db.add_documents(chunk)"""

    def add_documents_chunked(self, splitted_docs):
        # Prepare data for insertion
        texts = [doc.page_content for doc in splitted_docs]
        embeddings = self.embed_model.embed_documents(texts)
        data = [
            [i for i in range(len(texts))],  # You can use your own primary keys if needed
            embeddings,
            texts
        ]

        add_documents_chunked

        # Insert data into Milvus
        self.collection.insert(data)
        self.collection.flush()

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
