from typing import TypedDict, List, Annotated, Optional
from operator import add
from objects import AnswerQuestionObject, ReviseAnswerObject


class ReflexionState(TypedDict):
    task: str  # the original question being researched
    draft: Optional[AnswerQuestionObject]  # first-pass answer + self-critique + search queries
    search_results: Annotated[List[str], add]  # grows every time execute_tools_node runs
    revised: Optional[ReviseAnswerObject]  # latest revised answer with citations
    iteration: int  # how many draft/revise rounds have happened so far
