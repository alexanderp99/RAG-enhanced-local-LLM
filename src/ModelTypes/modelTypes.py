from enum import Enum
import subprocess


class Modeltype(Enum):
    LLAMA3_1_8B = "llama3.1:8b"
    AYA = "aya:latest"
    LLAMA3_2_1B = "llama3.2:1b"


def get_function_calling_modelfiles():
    modelfiles = []
    for model in Modeltype:
        if model != Modeltype.AYA:
            modelfiles.append(model.value)
    return modelfiles


def install_models():
    for model in Modeltype:
        try:
            subprocess.run(["ollama", "pull", model.value], check=True)
            print(f"installed {model.value}")
        except subprocess.CalledProcessError as e:
            print(f"error with {model.value}: {e}")


if __name__ == "__main__":
    install_models()
