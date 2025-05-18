from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

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

prompt1 = PromptTemplate(
    template="Please make a short and clear notes from the below text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate a few short question and answers from the below text \n {text}",
    input_variables=['text']
)

merge_prompt = PromptTemplate(
    template="Merge the both the notes and quiz into a single document within 256 tokens.\n notes -> {notes} \n quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parallel_chain = RunnableParallel({
    'notes' : prompt1 | llm | parser,
    'quiz' : prompt2 | llm | parser,
})

merge_chain = merge_prompt | llm | parser

# chain = parallel_chain | merge_prompt | llm | parser
chain = parallel_chain | merge_chain


text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke(input={'text': text})

print(result)

# chain.get_graph().print_ascii()



