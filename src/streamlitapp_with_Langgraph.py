import os

import pandas as pd
import streamlit as st

from src.LanggraphLLM import Langgraph

os.write(1, b'Logging...\n')
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


# Cache the Langgraph initialization
@st.cache(allow_output_mutation=True)
def initialize_langgraph():
    return Langgraph()


# Initialize Langgraph class only once
agent = initialize_langgraph()
document_vector_storage = agent.vectordb

st.title("Bachelor RAG local LLM")

selected_tab = st.sidebar.selectbox("Select Tab", ["Default", "vectordb"])

if selected_tab == "Default":
    uploaded_documents = []

    uploaded_file = st.file_uploader("Upload a file")

    view_messages = st.expander("View the message contents in session state")

    if uploaded_file is not None:
        uploaded_documents.append(uploaded_file)

        document_vector_storage.process_and_index_file(uploaded_file)

    if uploaded_documents:
        st.write("Uploaded Documents:")
        for document in uploaded_documents:
            st.write(document)

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")


    def respond_with_llm(user_input):
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = []

        st.session_state.conversation_memory.append({'role': 'user', 'content': user_input})

        os.write(1, f"{st.session_state.conversation_memory}\n".encode())

        latest_user_message_content = st.session_state.conversation_memory[-1]["content"]

        msgs.add_user_message(latest_user_message_content)

        # response = ChatOllama(model='llama3:instruct').invoke({"messages": [HumanMessage(content=latest_user_message_content)]})
        response = ChatOllama(model='llama3:instruct').invoke(msgs.messages)

        # response = ollama.chat(model='llama3:instruct', messages=st.session_state.conversation_memory)

        os.write(1, f"response: {response}\n".encode())
        os.write(1, f"response: {type(response)}\n".encode())
        os.write(1, f"response: {response.content}\n".encode())

        # response = agent.run(st.session_state.conversation_memory)

        # st.session_state.conversation_memory.append({'role': 'assistant', 'content': response['message']['content']})
        st.session_state.conversation_memory.append({'role': 'assistant', 'content': response.content})
        msgs.add_ai_message(response.content)

        return response.content


    for message in st.session_state.get("conversation_memory", []):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_prompt = st.chat_input("Your question:")

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
        view_messages.json(st.session_state.langchain_messages)

elif selected_tab == "vectordb":
    st.write("Filenames and Chunks in Vectordb:")
    filenames = document_vector_storage.get_indexed_filenames()
    chunks = document_vector_storage.get_all_chunks()

    os.write(1, f"{filenames}\n".encode())

    st.write("Filenames:")
    filenames_df = pd.DataFrame({"Filenames": filenames})
    st.write(filenames_df)

    st.write("Chunks:")
    chunks_df = pd.DataFrame({"Chunks": chunks})
    st.write(chunks_df)
