import operator
from typing import TypedDict, Annotated, Sequence

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun
from langchain_core.utils.function_calling import convert_to_openai_function

tools = [OpenWeatherMapQueryRun()]

model = ChatOllama(model='gemma:2b')
functions = [convert_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)


def function_1(state):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}


import json

from langchain_core.messages import FunctionMessage
from langgraph.prebuilt import ToolExecutor
from langgraph.prebuilt import ToolInvocation

tool_executor = ToolExecutor(tools)


def function_2(state):
    messages = state['messages']
    last_message = messages[-1]  # this has the query we need to send to the tool provided by the agent

    parsed_tool_input = json.loads(last_message.additional_kwargs["function_call"]["arguments"])

    # We construct an ToolInvocation from the function_call and pass in the tool name and the expected str input for OpenWeatherMap tool
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=parsed_tool_input['__arg1'],
    )

    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)

    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)

    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


def where_to_go(state):
    messages = state['messages']
    last_message = messages[-1]

    if "function_call" in last_message.additional_kwargs:
        return "continue"
    else:
        return "end"


# from langgraph.graph import Graph, END

# workflow = Graph()

# Or you could import StateGraph and pass AgentState to it
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("agent", function_1)
workflow.add_node("tool", function_2)

# The conditional edge requires the following info below.
# First, we define the start node. We use `agent`.
# This means these are the edges taken after the `agent` node is called.
# Next, we pass in the function that will determine which node is called next, in our case where_to_go().

workflow.add_conditional_edges("agent", where_to_go, {  # Based on the return from where_to_go
    # If return is "continue" then we call the tool node.
    "continue": "tool",
    # Otherwise we finish. END is a special node marking that the graph should finish.
    "end": END
}
                               )

# We now add a normal edge from `tools` to `agent`.
# This means that if `tool` is called, then it has to call the 'agent' next.
workflow.add_edge('tool', 'agent')

# Basically, agent node has the option to call a tool node based on a condition,
# whereas tool node must call the agent in all cases based on this setup.

workflow.set_entry_point("agent")

app = workflow.compile()

from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the temperature in las vegas")]}
app.invoke(inputs)

inputs = {"messages": [HumanMessage(content="what is the temperature in las vegas")]}
for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")
