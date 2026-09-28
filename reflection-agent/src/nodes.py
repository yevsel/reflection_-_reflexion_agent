from src.states import MessageState
from src.chains import generation_chain, reflection_chain
from langchain_core.messages import HumanMessage

def generate_tweet_node(state:MessageState):
    tweet = generation_chain.invoke({"messages":state['messages']})
    return{
        "messages":[tweet]
    }

def reflect_tweet_node(state:MessageState):
    reflection = reflection_chain.invoke({"messages": state["messages"]})
    return{
        "messages":[HumanMessage(content=reflection.content)]
    }