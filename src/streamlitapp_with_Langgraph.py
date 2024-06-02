from logging import Logger

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
last_agent_message: None = None

# Initialize Langgraph class only once
agent: Langgraph = Langgraph.get_langgraph_instance()
document_vector_storage: DocumentVectorStorage = agent.vectordb

st.title("Bachelor RAG local LLM")

selected_tab: str | None = st.sidebar.selectbox("Select Tab", ["Default", "vectordb"])
enable_document_search_button = st.toggle("Enable Document Search for every message")

if selected_tab == "Default":
    uploaded_document_names: list[str] = []

    if enable_document_search_button:
        agent.allow_document_search = True

    uploaded_file: list[UploadedFile] | None | UploadedFile = st.file_uploader("Upload a file",
                                                                               type=['txt', 'pdf', 'json', 'html',
                                                                                     "md"],
                                                                               accept_multiple_files=False)

    view_messages: DeltaGenerator = st.expander(
        "View the message contents in session state | Planned: How model arrived at last answer")

    if uploaded_file is not None:
        uploaded_document_names.append(uploaded_file.name)

        document_vector_storage.process_and_index_file(uploaded_file)

    if uploaded_document_names:
        st.write("Uploaded Documents:")
        for document_name in uploaded_document_names:
            st.write(document_name)

    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")


    def respond_with_llm(user_input: str):

        # latest_user_message_content = st.session_state.conversation_memory[-1]["content"]

        msgs.add_user_message(user_input)

        response_stream, final_response = agent.run_stream({"messages": msgs.messages})

        global last_agent_message
        last_agent_message = response_stream

        msgs.add_ai_message(final_response)

        return final_response


    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    user_prompt: str = st.chat_input("Your question:")

    if user_prompt:
        with st.chat_message("user"):
            st.write(user_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Loading..."):
                llm_response = respond_with_llm(user_prompt)
                st.write(llm_response)

    # Draw the messages at the end, so newly generated ones show up immediately
    with view_messages:
        """
        Contents of `st.session_state.langchain_messages`:
        """
        # view_messages.json(st.session_state.langchain_messages)
        st.write(last_agent_message)

elif selected_tab == "vectordb":

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
    st.dataframe(chunks_df, width=1100, height=800)
