from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# from dotenv import load_dotenv
# from huggingface_hub import login

import os
import torch

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
    max_new_tokens=256,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()
