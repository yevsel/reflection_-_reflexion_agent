from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage



class MessageState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]