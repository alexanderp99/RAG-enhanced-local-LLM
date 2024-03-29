import ollama
import streamlit as st

st.title("Bachelor RAG local LLM")

uploaded_documents = []

uploaded_file = st.file_uploader("Upload a file")

if uploaded_file is not None:
    uploaded_documents.append(uploaded_file)

if uploaded_documents:
    st.write("Uploaded Documents:")
    for document in uploaded_documents:
        st.write(document)


def respond_with_llm(user_input):
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []

    st.session_state.conversation_memory.append({'role': 'user', 'content': user_input})

    response = ollama.chat(model='gemma:2b', messages=st.session_state.conversation_memory)

    st.session_state.conversation_memory.append({'role': 'assistant', 'content': response['message']['content']})

    return response['message']['content']


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
