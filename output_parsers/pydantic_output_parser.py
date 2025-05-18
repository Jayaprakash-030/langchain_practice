from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from huggingface_hub import login
import os
import torch

# load_dotenv(dotenv_path="../.env")
# token = os.getenv("HF_TOKEN")
# login(token=token)

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

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate a fictional {place} person.\n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | llm | parser

result = chain.invoke({"place": "Indian"})
print(result)
print(type(result))