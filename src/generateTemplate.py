from langchain_core.messages import SystemMessage, HumanMessage


def generate_template(system_message, messages, tools_json):
    template = f"<|start_header_id|>system<|end_header_id|>\n\n"
    template += f"{system_message.get('content', '')}\n\n"

    if tools_json:
        template += "When you receive a tool call response, use the output to format an answer to the original user question.\n\n"
        template += "You are a helpful assistant with tool calling capabilities.\n\n"
        for tool in tools_json:
            template += f"{tool}\n"

    template += "<|eot_id|>\n"

    for message in messages:
        role = message['role']
        template += f"<|start_header_id|>{role}<|end_header_id|>\n"

        if role == "user" and tools_json and messages[-1] == message:
            template += "\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n"
            for tool in tools_json:
                template += f"{tool}\n"
            template += f"{message['content']}\n<|eot_id|>\n"
        else:
            template += f"\n{message['content']}\n<|eot_id|>\n"

    return template


print(
    generate_template(SystemMessage("You are a helpfull assistant."), messages=[HumanMessage("Why is the sky blue?")]))
