from typing import Union, Any, Literal

from langchain_core.messages import AnyMessage, HumanMessage
from pydantic import BaseModel


def reasoner(state, llm_with_tools, sys_message):
    messages = state["messages"]
    no_human_message_added = not (any([isinstance(each_message, HumanMessage) for each_message in messages]))
    sys_msg = sys_message
    message = state["question"]
    user_language_was_set = "user_language" in state and (state["user_language"] is not None)
    if user_language_was_set and message.content not in [msg.content for msg in messages]:
        messages.append(message)

    result = [llm_with_tools.invoke([sys_msg] + messages)]

    return {"messages": result}


def tools_condition(
        state: Union[list[AnyMessage], dict[str, Any], BaseModel],
        messages_key: str = "messages",
) -> Literal["tools", "EndNode"]:
    """Use in the conditional_edge to route to the ToolNode if the last message"""
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "EndNode"
