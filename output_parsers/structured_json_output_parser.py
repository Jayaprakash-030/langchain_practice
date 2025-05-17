from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch

# Step 1: Authenticate
load_dotenv(dotenv_path="../.env")
token = os.getenv("HF_TOKEN")
login(token=token)

# Step 2: Load LLaMA 2 model
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=token
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

# Step 3: Define the expected JSON schema
schema = [
    ResponseSchema(name="fact1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact3", description="Fact 3 about the topic"),
]
parser = StructuredOutputParser.from_response_schemas(schema)

# Step 4: Write a **natural-language prompt** without [INST] or <<SYS>>
prompt = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# Step 5: Build the chain
chain = prompt | llm

# Step 6: Run it
result = chain.invoke({"topic": "cricket"})
print(result)