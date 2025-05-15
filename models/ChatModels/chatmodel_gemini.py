from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

result = gemini.invole("Hey, How are you doing")

print(result.content)