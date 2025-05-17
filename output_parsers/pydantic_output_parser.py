from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv(dotenv_path="../.env")
token = os.getenv("HF_TOKEN")
login(token=token)


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "Generate a fictional {place} person.\n"
        "Return only a JSON object"
        "{format_instruction}"
    ),
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
    device=0,
    model_kwargs={
        "torch_dtype": "auto"
    },
    pipeline_kwargs={
        "temperature": 0.7,
        "pad_token_id": 128001,
        "do_sample": True,
        "return_full_text": False
    }
)

chain = template | llm | parser

result = chain.invoke({"place": "Indian"})
print(result)