from enum import Enum

# Source: https://ollama.com/library/gemma:latest
"gemma:7b"
"gemma:2b"
# "gemma-instruct"  # 9B
"gemma"  # 9b
"llama2:13b"


class Modeltype(Enum):
    LLAMA3_1_8B = "llama3.1:8b"
    LLAMA3_2_3B = "llama3.2:3b"
    LLAMA3_2_1B = "llama3.2:1b"
    MISTRAL_7B = "mistral:7b"
