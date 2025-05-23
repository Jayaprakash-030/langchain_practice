from langchain_community.document_loaders import PyPDFLoader # there are so many loaders as well.


loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)