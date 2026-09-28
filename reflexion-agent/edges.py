from langgraph.graph import END
from states import ReflexionState

MAX_ITERATIONS = 2  # how many draft/revise rounds to allow before stopping


def should_continue(state: ReflexionState):
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return END
    return "execute_tools_node"
