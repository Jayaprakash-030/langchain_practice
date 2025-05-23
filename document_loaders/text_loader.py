from langchain_community.document_loaders import TextLoader

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(type(docs)) # always will be a list of page_content, metadata

# print(len(docs))

# print(docs[0])

print(docs[0].page_content)
# print(docs[0].metadata)


from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    template= 'Explain the below poem in simple words within 150 words of text \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

chain = prompt | llm | parser

if docs:
    result = chain.invoke(input={'poem': docs[0].page_content})
    print(result)
else:
    print("No documents loaded from the file.")
