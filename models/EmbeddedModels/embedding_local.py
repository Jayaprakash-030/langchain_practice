from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France",
]

vector = embeddings.embed_documents(documents)

array = np.array(vector)
print(array.shape) # (3,384) -> (no of sentences, vector dimesion)
print(str(vector))
