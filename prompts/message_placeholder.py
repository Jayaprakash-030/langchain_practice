from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system","You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{query}")
])

chat_history = []

with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

# print(chat_history)

prompt = chat_template.invoke({"query":"Where is my refund","chat_history":chat_history})

print(prompt)