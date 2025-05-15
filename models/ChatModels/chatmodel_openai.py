from langchain_openai import ChatOpenAI # it inherits from base chat openai
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=1.5, max_completion_tokens=50) # you can chnage your temperature for creative work and other works
# max_completion_tokens will reduce the no of tokens in the input
result = llm.invoke("What is your name")

print("result")