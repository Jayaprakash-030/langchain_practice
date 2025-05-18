from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

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

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] =  Field(description='Classify the sentiment into either postive or negative')
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="Classify the below text into a postive or negative feedback. \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

prompt1 = PromptTemplate(
    template='Write a appropriate response to this postive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt2 = PromptTemplate(
    template='Write a appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

feedback = "S24 ultra is a good phone"

chain = prompt | llm | parser2

branch_chain = RunnableBranch(
    (RunnableLambda(lambda x : x.sentiment == 'positive') , prompt1 | llm | parser),
    (RunnableLambda(lambda x : x.sentiment == 'negative'), prompt2 | llm | parser),
    RunnablePassthrough(lambda x : "could not find the sentiment")
)

final_chain = chain | branch_chain

result = final_chain.invoke(input={'feedback':feedback})
print(result)