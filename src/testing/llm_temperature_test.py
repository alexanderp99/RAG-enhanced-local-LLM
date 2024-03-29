from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="gemma", temperature=0)
messages = [
    HumanMessage(
        content="What color is the sky at different times of the day? "
    )
]
print(llm.invoke(messages).content)
