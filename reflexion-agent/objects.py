from pydantic import BaseModel, Field
from typing import List


class ReflectionObject(BaseModel):
    missing: str = Field(description="A critique of what is missing.")
    unnecessary_information: str = Field(description="A critique of unnecessary information.")
    
    
class AnswerQuestionObject(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: str = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")


class ReviseAnswerObject(AnswerQuestionObject):
    references: List[str] = Field(description="Citations motivating your updated answer, as URLs.")