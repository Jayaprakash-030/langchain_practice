from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=64,
    do_sample=True,
    top_p=0.95,
    return_full_text=False,
)

def format_prompt(user_input):
    return f"### Instruction:\n{user_input}\n\n### Response:"

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    else:
        prompt = format_prompt(user_input=user_input)
        result = llm.invoke(prompt)
        print("AI:", result)
