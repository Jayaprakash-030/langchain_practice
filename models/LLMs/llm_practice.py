from langchain_openai import OpenAI # it inherits from base openai
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")
result = llm.invoke("What is favourite cricketing ground in India")

print(result)

# llms are not used that much now. Use chatmodels instead.