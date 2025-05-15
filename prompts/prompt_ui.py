from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFacePipeline,
)
from dotenv import load_dotenv
import streamlit as st

load_dotenv(dotenv_path="../models/.env")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation"
)

st.header("Research Tool")

user_input = st.text_input("Enter your prompt")  # static prompt

if st.button("Summarize"):
    result = llm.invoke(user_input)
    st.write(result)
