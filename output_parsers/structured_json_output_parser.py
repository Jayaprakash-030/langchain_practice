from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
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

schema = [
    ResponseSchema(name="fact1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact3", description="Fact 3 about the topic"),
]
parser = StructuredOutputParser.from_response_schemas(schema)

prompt = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = prompt | llm | parser

result = chain.invoke({"topic": "cricket"})
print(result)

# this output parser cannot validate data.