# Installation

I am assuming that python is already installed.

First we have to install all the requirements:
> pip install -r requirements.txt

Then we fetch an LLM from ollama

> ollama pull 'any_model'
>
> ollama pull gemma

# Usage

We start the application with:
> python -m streamlit run ./src/streamlitapp.py

# Run Unit Tests

> python -m unittest ?

# Lessons Learned

Streamlit Ausführen IDE
Scriptpath: <python.exe>
Script parameters: -m streamlit.web.cli run

Streamlit Debuggen IDE:
Scriptpath(streamlit!): C:/Users/alexp/PycharmProjects/RAG-enhanced-local-LLM/.venv/Lib/site-packages/streamlit
Script parameters: run simple_streamlit.py
Working directory: C:\Users\alexp\PycharmProjects\RAG-enhanced-local-LLM\src

- Für die Python Version 3.12 funktioniert der Debugger bei Chroma db nicht. Es muss 3.11 verwendet werden
