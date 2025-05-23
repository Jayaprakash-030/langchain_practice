from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader( # we can chnage the order as well, but no attribute is present in the class but we can do it
    path="books",
    glob='*.pdf',
    loader_cls= PyPDFLoader
)

docs = loader.load()

lazy_docs = loader.lazy_load()

# print(len(docs))
# print(docs[1].page_content)
# print(docs[0].metadata)

# print(docs[326].page_content)
# print(docs[326].metadata)

# for document in docs: # this will only print after its load every page of all the pdfs in the directory
#     print(document.metadata) 

for document in lazy_docs: # this will load one document at a time and print its metadata and then deletes that from the memory
    print(document.metadata)