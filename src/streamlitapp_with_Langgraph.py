from typing import Any

import pandas as pd
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from pandas import DataFrame
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.VectorDatabase import DocumentVectorStorage
from src.ReasoningLanggraphLLM import ReasoningLanggraphLLM
from src.ModelTypes.modelTypes import get_function_calling_modelfiles

st.set_page_config(layout="wide")
session_key: str = "langchain_messages"
msgs: StreamlitChatMessageHistory = StreamlitChatMessageHistory(key=session_key)
last_response_stream: Any = None

# Initialize Langgraph class only once
agent: ReasoningLanggraphLLM = ReasoningLanggraphLLM.get_langgraph_instance()
document_vector_storage: DocumentVectorStorage = agent.vectordb

selected_tab: str | None = st.sidebar.selectbox("Select Tab", ["Default", "vectordb", "settings"])
greeting_message: str = "How can I help you?"


def on_model_change():
    selected_value = st.session_state["model_selection"]
    agent.change_selected_model(selected_value)


def on_profanity_button_change():
    enable_profanity_check: bool = st.session_state["profanity_button_selection"]
    agent.change_profanity_check(enable_profanity_check)


def clear_langgraph_conversation():
    agent.reset_memory()
    msgs.clear()
    st.session_state.reset_langgraph_cache_button = False


if "reset_langgraph_cache_button" not in st.session_state:
    st.session_state.reset_langgraph_cache_button = False

if "allow_profanity_check" not in st.session_state:
    st.session_state.allow_profanity_check = False

if selected_tab == "Default":
    uploaded_document_names: list[str] = []

    st.button("Reset Conversation Memory", on_click=clear_langgraph_conversation)

    uploaded_file: list[UploadedFile] | None | UploadedFile = st.file_uploader(
        "A file here is added to the vector database and used for text retrieval",
        type=['txt', 'pdf', 'json',
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
        msgs.add_ai_message(greeting_message)


    def respond_with_llm():
        messages_without_greeting = [message for message in msgs.messages if message.content != greeting_message]
        response_stream, final_response = agent.run_stream({"messages": messages_without_greeting})

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

            full_message = llm_response

            msgs.add_ai_message(full_message)

    # Show the response stream in expander
    with view_messages:
        st.text(last_response_stream)

elif selected_tab == "vectordb":
    if st.button("Empty Vector Database", type="primary"):
        document_vector_storage.remove_all_documents()

    st.write("Filenames and Chunks in Vectordb:")

    filenames = document_vector_storage.get_indexed_filenames()
    chunks, repeated_list_filenames, summaries = document_vector_storage.get_all_chunks()

    st.write("Filenames:")
    if filenames:
        filenames_df: DataFrame = pd.DataFrame({"Filenames": filenames})
    else:
        filenames_df = pd.DataFrame(columns=["Filenames"])
    st.dataframe(filenames_df, width=1100)

    st.write("Chunks:")
    if chunks:
        chunks_df: DataFrame = pd.DataFrame(
            {"Chunks": chunks, "Filename": repeated_list_filenames, "Summary": summaries})
    else:
        chunks_df = pd.DataFrame(columns=["Chunks"])
    st.dataframe(chunks_df, width=1100)
elif selected_tab == "settings":
    option = st.selectbox(
        "Which LLM should be used for function calling/ profanity check",
        options=get_function_calling_modelfiles(), index=0, on_change=on_model_change, key="model_selection",
    )
    st.session_state.allow_profanity_check = st.toggle(
        "Enable Profanity Check", value=st.session_state.allow_profanity_check, on_change=on_profanity_button_change,
        key="profanity_button_selection", )
