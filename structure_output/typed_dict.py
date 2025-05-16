from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
from huggingface_hub import login
from typing import TypedDict

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
    max_new_tokens=64,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)


#schema

class Review(TypedDict):

    summary:str
    sentiment:str

structured_output = llm.with_structured_output(Review)

result = structured_output.invoke('''
I recently tried the new electric SUV from Tesla, and I have mixed feelings about it. The acceleration and handling were absolutely top-notch — it feels like driving the future. However, the build quality on the interior was underwhelming for the price point. There were some rattling noises, and the infotainment system froze once during my test drive. Overall, it's a great step forward for EVs, but there’s room for improvement.
''')

print(result)