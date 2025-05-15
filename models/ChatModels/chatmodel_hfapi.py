from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

# Define the LLM using Hugging Face Inference API
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)

prompt = "Explain the difference between AI, Machine Learning, and Deep Learning."
response = llm.invoke(prompt)

print(response)