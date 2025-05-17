from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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
    max_new_tokens=128,
)

llm = HuggingFacePipeline(pipeline=pipe)

parser = JsonOutputParser() # cannot have a schema

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person. Respond ONLY with a JSON object. No explanation or extra text. \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# without chains
# prompt = template.format()
# result = llm.invoke(prompt)
# print(result)
# print('\n')
# print(parser.parse(result))

chain = template | llm | parser

result = chain.invoke(input={})

print(result)
