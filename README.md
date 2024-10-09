# Installation

## Pre installation

### Python 3.12.7 - Oct. 1, 2024 installation

Due to the removal of the long-deprecated pkgutil.ImpImporter class, the pip command may not work for Python 3.12 out of
the box
> python -m ensurepip --upgrade
> python -m pip install --upgrade setuptools

###

Big Pc setup
Pip: 24.2
Python: 3.11.8
Ollama: 0.3.11

### Ubuntu

On ubuntu you need to install python venv
> sudo apt install python3.11-venv

Create a virtual env
> python3 -m venv .venv

Activate it
> source .venv/bin/activate

I am assuming

- python is already installed.
- CUDA is already installed and working

Recommended python Version: 3.11.10

First we have to install all the requirements:
> pip install -r requirements.txt

upgrade pip
> python -m pip install --upgrade pip

or update if we did not use the project for a longer time :
> pip install -r requirements.txt --upgrade

We can install the software ollama on linux like this
> curl -fsSL https://ollama.com/install.sh | sh

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
Scriptpath(streamlit!): C:/Users/alexp/PycharmProjects/RAG-enhanceollama stopd-local-LLM/.venv/Lib/site-packages/streamlit
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

## Generating UML File

You can generate UML files (.puml) with the command if your current directory is the 'src' directory with:
> pyreverse -o puml LanggraphLLM.py -d ./src_uml

Or if you want an image you can do in the 'src' directory:
> pyreverse LanggraphLLM.py -d ./src_uml
> cd src_uml
> dot -Tpng classes.dot -o classes.png