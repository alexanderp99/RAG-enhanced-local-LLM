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