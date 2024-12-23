from typing import List

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
).bind_tools([validate_user])

result = llm.invoke(
    "Why is the sky blue?"
)
print(result)
print(result.tool_calls)
