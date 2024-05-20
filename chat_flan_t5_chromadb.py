# -*- coding: utf-8 -*-
"""chat_flan-t5_chromadb.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19T57OGLFwUu84okFpphuebRETZsEeU59
"""
import streamlit as st



from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import notebook_login
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import sys
import numpy as np
import faiss

!mkdir docs

document = []
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        document.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./docs/" + file
        loader = Docx2txtLoader(doc_path)
        document.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./docs/" + file
        loader = TextLoader(text_path)
        document.extend(loader.load())

document_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
document_chunks = document_splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
document_texts = [doc.page_content for doc in document_chunks]
document_embeddings = embeddings.embed_documents(document_texts)
document_embeddings_array = np.array(document_embeddings)

# Build the FAISS index
d = document_embeddings_array.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
index.add(document_embeddings_array)

from langchain.vectorstores import Chroma
vectordb=Chroma.from_documents(document_chunks,embedding=embeddings, persist_directory='./data')

vectordb

vectordb.persist()

notebook_login()

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large",
                                              device_map='auto',
                                              torch_dtype=torch.float16)

pipe = pipeline("text2text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map='auto',
                max_length=512)

llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

retriever = vectordb.as_retriever(search_kwargs={'k': 6})
pdf_qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                               retriever=retriever,
                                               verbose=False,
                                               memory=memory)

'''print('---------------------------------------------------------------------------------')
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')

while True:
  query=input(f"Prompt:")
  if query == "exit" or query == "quit" or query == "q" or query == "f":
    print('Exiting')
    sys.exit()
  if query == '':
    continue
  result = pdf_qa({"question": query})
  print(f"Answer: " + result["answer"])
'''

print('---------------------------------------------------------------------------------')
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"Prompt:")
    if query.lower() in ["exit", "quit", "q", "f"]:
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    # Perform FAISS similarity search
    query_embedding = np.array(embeddings.embed_documents([query])).reshape(1, -1)
    k = 6
    distances, indices = index.search(query_embedding, k)
    # Retrieve the documents and their scores
    results = [(document_chunks[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    for doc, score in results:
        print(f"Document: {doc.page_content}")
        print(f"Score: {score}")
    top_document = results[0][0].page_content if results else "No relevant documents found."
    result = pdf_qa({"question": query, "context": top_document})
    print(f"Answer: {result['answer']}")



