import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from objects import AnswerQuestionObject, ReflectionObject
from langchain_core.messages import SystemMessage, HumanMessage

# LLM
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

# Structured output LLM — forces responses to come back shaped like AnswerQuestionObject
structured_llm = llm.with_structured_output(AnswerQuestionObject)

current_time = lambda : datetime.datetime.now().isoformat()

# Prompts
actor_prompt = """You are an expert researcher. Current time: {time} 
    1. {first_instruction}
    2. Reflect and critique your answer. Be severe to maximize improvement.
    3. Recommend search queries to research information and improve your answer.
"""

first_responder_prompt = actor_prompt.format(time=current_time(),first_instruction="Provide a detailed ~250 word answer")
first_responder_system_message = SystemMessage(content=first_responder_prompt)

human_message = HumanMessage(content="Write about AI-powered SOC/autonomous soc problem domain, list startups that do that and raised capital")

first_responder_response = structured_llm.invoke([first_responder_system_message, human_message])

# print(first_responder_response)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not count towards the new limit). In form of:
        - [1] https://example.com
        - [2] https://example.com
    - You should use the previous critique to remove unnecessary information from your answer and make SURE it is not more than 250 words
"""