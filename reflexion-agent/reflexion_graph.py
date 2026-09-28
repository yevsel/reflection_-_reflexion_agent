from langgraph.graph import StateGraph, START, END

from states import ReflexionState
from nodes import draft_node, execute_tools_node, revise_node
from edges import should_continue

builder = StateGraph(ReflexionState)
builder.add_node("draft_node", draft_node)
builder.add_node("execute_tools_node", execute_tools_node)
builder.add_node("revise_node", revise_node)

builder.add_edge(START, "draft_node")
builder.add_edge("draft_node", "execute_tools_node")
builder.add_edge("execute_tools_node", "revise_node")
builder.add_conditional_edges(
    "revise_node",
    should_continue,
    {"execute_tools_node": "execute_tools_node", END: END},
)

graph = builder.compile()
