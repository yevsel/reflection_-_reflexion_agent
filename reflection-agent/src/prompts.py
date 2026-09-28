from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the users tweet. Always provide detailed recommendations, including requests for length, virality, style etc"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a twitter techie influencer assistant tasked with writing excellent twitter posts. Generate the best twitter post possible for the users request. If the user provides a critique, respond with a revised version of your previous attempts."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

