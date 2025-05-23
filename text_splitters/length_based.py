from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="../document_loaders/dl-curriculum.pdf") #document_loaders/dl-curriculum.pdf

docs = loader.load()

splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size = 200,
    chunk_overlap=20,
)

result = splitter.split_documents(docs)

print(result)


# long_text = """This is a long piece of text designed to demonstrate the CharacterTextSplitter.
# It has multiple sentences and multiple lines. We want to split this into chunks.
# The goal is to see how the splitter works with a defined chunk size.

# New paragraphs are often good places to split, but the CharacterTextSplitter will also
# split mid-sentence if necessary to meet the chunk size requirement, trying to do so
# at spaces or newlines first. Let's add more content to ensure we exceed the chunk size
# multiple times. This splitter is useful for simple, character-based splitting tasks
# when more sophisticated semantic splitting is not required or is too complex.
# It's a fundamental tool in preparing documents for language models which often have
# context window limitations. We are aiming for a chunk size of 200 characters.
# This means each chunk should be at most 200 characters long. The overlap can also be set.
# Repeating some content for length: Lorem ipsum dolor sit amet, consectetur adipiscing elit,
# sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
# quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
# Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
# Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
# Hopefully, this is enough text to show several chunks.
# """


# result = splitter.split_text(long_text)

# print(result)

# for i, chunk in enumerate(result):
#     print(f"--- Chunk {i+1} (Length: {len(chunk)}) ---")
#     print(chunk)
#     print("\n")

