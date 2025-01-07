from enum import Enum
import subprocess


class Modeltype(Enum):
    LLAMA3_1_8B = "llama3.1:8b"
    MISTRAL_7B = "mistral:7b"
    AYA = "aya:latest"


def install_models():
    for model in Modeltype:
        try:
            subprocess.run(["ollama", "pull", model.value], check=True)
            print(f"installed {model.value}")
        except subprocess.CalledProcessError as e:
            print(f"error with {model.value}: {e}")


if __name__ == "__main__":
    install_models()
