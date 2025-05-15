from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

result = embedding.embed_query("Hi, My name is Jayaprakash")

print(str(result)) ## str just to see them i a proper format


# Now lets see when we have multiple sentences i.e. documents how can we do it

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France" 
]

result_doc = embedding.embed_documents(documents) # we have used just embed documents instead of embed query

print(str(result_doc))