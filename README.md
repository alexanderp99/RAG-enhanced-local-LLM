# Installation

I am assuming

- python is already installed.
- CUDA is already installed and working

Recommended python Version: 3.11.10

First we have to install all the requirements:
> pip install -r requirements.txt

Then we fetch the LLM from ollama

> ollama pull 'llama3:instruct'

# Usage

Current directory: Project directory

We start the application with:
> python -m streamlit run ./src/streamlitapp_with_Langgraph.py

# Features

- Normale Reponse
- User Input Profoundness Check
- Give answer using Document retrieval
- Hallucination Check

## Vector Database

- The Vector Database is persistent. It is stored under the folder /chroma_db.
- The Vector Database makes sure that the same file is only added once (determined by filename).

## File Indexing

- You can index files via 2 ways

1. All files within the folder /indexedFiles are automatically added to the vector database.
2. Via UI in the app. The uploaded file is also saved in the /indexedFiles directory

- You can empty the database

# Run Unit Tests

Current directory: Project directory
> python -m unittest discover -s tests -p '*.py'

# Lessons Learned

Streamlit Ausführen IDE
Scriptpath: <python.exe>
Script parameters: -m streamlit.web.cli run

Streamlit Debuggen IDE:
Scriptpath(streamlit!): C:/Users/alexp/PycharmProjects/RAG-enhanced-local-LLM/.venv/Lib/site-packages/streamlit
Script parameters: run ./src/simple_streamlit.py
Working directory: C:\Users\alexp\PycharmProjects\RAG-enhanced-local-LLM

Hotreload kann über Menü(...) rechts oben eingestellt werden

- Für die Python Version 3.12 funktioniert der Debugger bei Chroma db nicht. Es muss 3.11 verwendet werden
- Langchain ist manchmal eher eine Einschränkung als Hilfe
- Streamlit Langchain Integration
- Datenverarbeitung (text splitting)
- App Startzeit ist schwer zu optimieren
- Hohe startzeit macht debugging deutlich anstrengender
- Prompting ist eine Kunst
- Function Calling ist eine Kunst
- Schwer einen Überblick über alle Modelle zu bekommen