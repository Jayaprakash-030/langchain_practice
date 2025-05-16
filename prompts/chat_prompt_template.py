from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

chat_template = ChatPromptTemplate([      # we can also use ChatPromptTemplate.from_messages
    ("system", "You are a helpful {domain} expert"),
    ("human", "Explain in simple terms what is {topic}")

    # SystemMessage(content="You are a helpful {domain} expert"),
    # HumanMessage(content="Explain in simple terms what is {topic}")
])

prompt = chat_template.invoke({"domain":"cricket", "topic":"duck"})

print(prompt)