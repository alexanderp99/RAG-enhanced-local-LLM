from operator import itemgetter

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
# Set up the LLM which will power our application.
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="ollama",
    temperature=0,
    model="llama3:instruct",
    base_url="http://localhost:11434/v1",
)
from langchain_community.tools import DuckDuckGoSearchResults

# from src.VectorDatabase import DocumentVectorStorage
#
# vectordb = DocumentVectorStorage()

# , vectordb.get_local_knowledge
tools = [DuckDuckGoSearchResults()]
model = llm.bind_tools(tools)


# Define tools available.
@tool
def multiply(first: int, second: int) -> int:
    """Multiply two integers together."""
    return first * second


@tool
def add(first: int, second: int) -> int:
    "Add two integers."
    return first + second


@tool
def converse(input: str) -> str:
    "Provide a natural language response using the user input."
    return model.invoke(input)


tools = [add, multiply, converse, DuckDuckGoSearchResults()]

# Configure the system prompts
# rendered_tools = render_text_description(tools)
from langchain_core.utils.function_calling import convert_to_openai_function

rendered_tools = [convert_to_openai_function(t) for t in tools]
system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys. The value associated with the 'arguments' key should be a dictionary of parameters."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)


# Define a function which returns the chosen tools as a runnable, based on user input.
def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool


# The main chain: an LLM with tools.
chain = prompt | model | JsonOutputParser() | tool_chain

# Set up message history.
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("I can add, multiply, or just chat! How can I help you?")

# Set the page title.
st.title("Chatbot with tools")

# Render the chat history.
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# React to user input
if input := st.chat_input("What is up?"):
    # Display user input and save to message history.
    st.chat_message("user").write(input)
    msgs.add_user_message(input)

    # Invoke chain to get reponse.
    response = chain.invoke({'input': input})

    # Display AI assistant response and save to message history.
    st.chat_message("assistant").write(str(response))
    msgs.add_ai_message(response)
