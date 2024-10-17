from langchain_community.tools import DuckDuckGoSearchResults

tool = DuckDuckGoSearchResults()
tool.invoke("Obama")
