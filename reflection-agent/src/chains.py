from dotenv import load_dotenv
load_dotenv()

from src.llm_models import llm
from src.prompts import reflection_prompt, generation_prompt


generation_chain = generation_prompt | llm

reflection_chain = reflection_prompt | llm