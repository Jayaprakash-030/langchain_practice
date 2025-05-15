from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()


pipe = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",
    tokenizer="tiiuae/falcon-rw-1b",
    max_new_tokens=64,
    temperature=0.2,
    top_p=0.95,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=pipe)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about Langchain")
]

result = llm.invoke(messages)

messages.append(AIMessage(content=result))

print(messages)