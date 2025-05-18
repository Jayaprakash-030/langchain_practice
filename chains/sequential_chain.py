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
    max_new_tokens=512,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    template='Generate a deatiled report about \n {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='PLease give 3 quiz questions for the below text and also give the detailed answer for them as well \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | llm | parser | prompt2 | llm | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()
