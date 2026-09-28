from langgraph.graph import StateGraph, START,END
from src.states import MessageState
from src.nodes import generate_tweet_node, reflect_tweet_node
from src.edges import should_continue

builder = StateGraph(MessageState)
builder.add_node("generate_tweet_node",generate_tweet_node)
builder.add_node("reflect_tweet_node",reflect_tweet_node)

builder.add_edge(START,"generate_tweet_node")
builder.add_edge("reflect_tweet_node","generate_tweet_node")
builder.add_conditional_edges("generate_tweet_node",should_continue,path_map={END:END,"reflect_tweet_node":"reflect_tweet_node"})

graph = builder.compile()
