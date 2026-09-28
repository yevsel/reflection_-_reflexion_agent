from langgraph.graph import END
from src.states import MessageState


def should_continue(state:MessageState):
    if len(state['messages']) > 6:
        return END
    return "reflect_tweet_node"