from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from huggingface_hub import login

import os
import torch

load_dotenv(dotenv_path="../.env")

token = os.getenv("HF_TOKEN")
login(token=token)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.2,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
    )

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n{text}",
    input_variables=['text']
    )

prompt1 = template1.invoke(input={"topic":"blackhole"})

result1 = llm.invoke(prompt1)

print(result1)

prompt2 = template2.invoke(input={"text":result1})

result2 = llm.invoke(prompt2)

print(result2)