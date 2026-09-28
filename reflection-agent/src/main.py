from typing import TypedDict,Annotated,List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()
# LLM
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

# State
class MessagesState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]

# Prompt
generate_tweet_prompt="""You are a twitter techie influencer assistant tasked with writing excellent twitter posts. Generate the best twitter post possible for the users request. If the user provides a critique, respond with a revised version of your previous attempts."""

reflect_tweet_prompt="""You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the users tweet. Always provide detailed recommendations, including requests for length, virality, style etc"""


# Nodes
def generate_tweet_node(state:MessagesState):
    generate_tweet_system_message = SystemMessage(content=generate_tweet_prompt)
    response = llm.invoke([generate_tweet_system_message]+state["messages"])
    return {"messages":[response]}

def reflect_tweet_node(state:MessagesState):
    reflect_tweet_system_message = SystemMessage(content=reflect_tweet_prompt)
    response = llm.invoke([reflect_tweet_system_message]+state["messages"])
    return {"messages":[HumanMessage(content=response.content)]}
    
    
# Edges
def should_continue_to_reflect(state:MessagesState):
    if len(state["messages"]) > 6:
        return END
    return "reflect_tweet_node"


# Builder
builder = StateGraph(MessagesState)
builder.add_node("generate_tweet_node",generate_tweet_node)
builder.add_node("reflect_tweet_node",reflect_tweet_node)

builder.add_edge(START,"generate_tweet_node")
builder.add_edge("reflect_tweet_node","generate_tweet_node")
builder.add_conditional_edges("generate_tweet_node",should_continue_to_reflect,path_map={END:END,"reflect_tweet_node":"reflect_tweet_node"})

graph = builder.compile()