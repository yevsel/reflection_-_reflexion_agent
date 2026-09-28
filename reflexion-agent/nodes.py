import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

from llm_models import responder_llm, revisor_llm
from prompts import actor_prompt, revise_instructions
from states import ReflexionState

current_time = lambda: datetime.datetime.now().isoformat()


def draft_node(state: ReflexionState):
    """First pass: answer the task, critique that answer, and suggest search queries to improve it."""
    prompt = actor_prompt.format(time=current_time(), first_instruction="Provide a detailed ~250 word answer")
    draft = responder_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["task"])])
    return {"draft": draft, "iteration": state.get("iteration", 0) + 1}


def execute_tools_node(state: ReflexionState):
    """Run each suggested search query through Tavily, collecting real sources for the revision step."""
    # After a revision, use its (possibly updated) search_queries; otherwise use the first draft's.
    latest = state.get("revised") or state["draft"]
    queries = latest.search_queries

    tavily = TavilySearch(max_results=3)
    results = []
    for query in queries:
        response = tavily.invoke(query)
        formatted = "\n\n".join(f"{doc['url']}: {doc['content']}" for doc in response["results"])
        results.append(f"Query: {query}\n{formatted}")

    return {"search_results": results}


def revise_node(state: ReflexionState):
    """Actor revises the answer using its own critique plus the new search results, and adds citations."""
    context = "\n\n---\n\n".join(state["search_results"])
    prompt = (
        f"{revise_instructions}\n\n"
        f"Original question: {state['task']}\n\n"
        f"Previous answer: {state['draft'].answer}\n\n"
        f"Previous critique: {state['draft'].reflection}\n\n"
        f"New information:\n{context}"
    )
    revised = revisor_llm.invoke([SystemMessage(content=prompt)])
    return {"revised": revised, "iteration": state.get("iteration", 0) + 1}
