from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from objects import AnswerQuestionObject, ReviseAnswerObject

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Forces the first draft to come back shaped like AnswerQuestionObject
responder_llm = llm.with_structured_output(AnswerQuestionObject)

# Forces the revision to come back shaped like ReviseAnswerObject (adds citations)
revisor_llm = llm.with_structured_output(ReviseAnswerObject)
