# Installation

I am assuming

- python is already installed.
- CUDA is already installed and working

Recommended python Version: 3.11.10

First we have to install all the requirements:
> pip install -r requirements.txt

Then we fetch an LLM from ollama

> ollama pull 'phi3'

# Usage

Current directory: Project directory

We start the application with:
> python -m streamlit run ./src/streamlitapp_with_Langgraph.py

# Run Unit Tests

Current directory: Project directory
> python -m unittest discover -s tests -p '*.py'

# Lessons Learned

Streamlit Ausführen IDE
Scriptpath: <python.exe>
Script parameters: -m streamlit.web.cli run

Streamlit Debuggen IDE:
Scriptpath(streamlit!): C:/Users/alexp/PycharmProjects/RAG-enhanced-local-LLM/.venv/Lib/site-packages/streamlit
Script parameters: run simple_streamlit.py
Working directory: C:\Users\alexp\PycharmProjects\RAG-enhanced-local-LLM\src

- Für die Python Version 3.12 funktioniert der Debugger bei Chroma db nicht. Es muss 3.11 verwendet werden
- Langchain ist manchmal eher eine Einschränkung als Hilfe
- Streamlit Langchain Integration
- Datenverarbeitung (text splitting)