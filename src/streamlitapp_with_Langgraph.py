from logging import Logger
from typing import Any

import pandas as pd
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from pandas import DataFrame
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.LanggraphLLM import Langgraph
from src.VectorDatabase import DocumentVectorStorage
from src.configuration.logger_config import setup_logging

logger: Logger = setup_logging()

st.set_page_config(layout="wide")
session_key: str = "langchain_messages"
msgs: StreamlitChatMessageHistory = StreamlitChatMessageHistory(key=session_key)
last_response_stream: Any = None

# Initialize Langgraph class only once
agent: Langgraph = Langgraph.get_langgraph_instance()
document_vector_storage: DocumentVectorStorage = agent.vectordb

st.title("Bachelor RAG local LLM")

selected_tab: str | None = st.sidebar.selectbox("Select Tab", ["Default", "vectordb"])

if "enable_document_search_button" not in st.session_state:
    st.session_state.enable_document_search_button = True
if "enable_profanity_check_button" not in st.session_state:
    st.session_state.enable_profanity_check_button = False
if "enable_hallucination_check_button" not in st.session_state:
    st.session_state.enable_hallucination_check_button = False

if selected_tab == "Default":
    uploaded_document_names: list[str] = []

    st.session_state.enable_document_search_button = st.toggle(
        "Enable Document Search", value=st.session_state.enable_document_search_button)

    agent.allow_document_search = st.session_state.enable_document_search_button
    agent.allow_profanity_check = st.session_state.enable_profanity_check_button
    agent.allow_hallucination_check = st.session_state.enable_hallucination_check_button

    uploaded_file: list[UploadedFile] | None | UploadedFile = st.file_uploader(
        "A file here is added to the vector database and used for text retrieval",
        type=['txt', 'pdf', 'json', 'html',
              "md"],
        accept_multiple_files=False)

    view_messages: DeltaGenerator = st.expander(
        "Press here to look how the agent arrived at the last answer (graph state)")

    if uploaded_file is not None:
        uploaded_document_names.append(uploaded_file.name)

        document_vector_storage.process_and_index_file(uploaded_file)

    if uploaded_document_names:
        st.write("Uploaded Documents:")
        for document_name in uploaded_document_names:
            st.write(document_name)

    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")


    def respond_with_llm():
        response_stream, final_response = agent.run_stream({"messages": msgs.messages})

        global last_response_stream
        last_response_stream = response_stream

        return final_response


    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    user_prompt: str = st.chat_input("Your question:")

    if user_prompt:
        with st.chat_message("user"):
            st.write(user_prompt)
        assitant_message = st.chat_message("assistant")
        with st.spinner("Loading..."):
            msgs.add_user_message(user_prompt)
            llm_response = respond_with_llm()
            assitant_message.write(llm_response)

            rag_context_state = agent.graph.get_state({"configurable": {"thread_id": "1"}}).values["rag_context"]
            if len(rag_context_state) > 0:
                rag_context_str = ''.join([item.page_content for item in rag_context_state])
                markdown_message = "\n\n:blue[Source]"
                assitant_message.markdown(markdown_message, help=rag_context_str)

                full_message = llm_response + markdown_message
            else:
                full_message = llm_response

            msgs.add_ai_message(full_message)

    # Show the response stream in expander
    with view_messages:
        st.text(last_response_stream)

elif selected_tab == "vectordb":

    st.session_state.enable_profanity_check_button = st.toggle(
        "Enable Profanity Check", value=st.session_state.enable_profanity_check_button)
    st.session_state.enable_hallucination_check_button = st.toggle(
        "Enable Hallucination Check", value=st.session_state.enable_hallucination_check_button)

    if st.button("Empty Vector Database", type="primary"):
        document_vector_storage.db.delete_collection()

    st.write("Filenames and Chunks in Vectordb:")

    filenames = document_vector_storage.get_indexed_filenames()
    chunks, repeated_list_filenames = document_vector_storage.get_all_chunks()

    st.write("Filenames:")
    if filenames:
        filenames_df: DataFrame = pd.DataFrame({"Filenames": filenames})
    else:
        filenames_df = pd.DataFrame(columns=["Filenames"])
    st.dataframe(filenames_df, width=1100)

    st.write("Chunks:")
    if chunks:
        chunks_df: DataFrame = pd.DataFrame({"Chunks": chunks, "Filename": repeated_list_filenames})
    else:
        chunks_df = pd.DataFrame(columns=["Chunks"])
    st.dataframe(chunks_df, width=1100)
