import streamlit as st
import os
import sys
import numpy as np
import faiss
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Create a directory for documents
if not os.path.exists('docs'):
    os.makedirs('docs')

# Streamlit title
st.title('Document Assistant')

# User input for document files
uploaded_files = st.file_uploader("Upload one or more documents", type=['pdf', 'docx', 'doc', 'txt'], accept_multiple_files=True)

document = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(uploaded_file)
            document.extend(loader.load())
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            loader = Docx2txtLoader(uploaded_file)
            document.extend(loader.load())
        elif uploaded_file.type == 'text/plain':
            loader = TextLoader(uploaded_file)
            document.extend(loader.load())

# Split documents into chunks
document_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
document_chunks = document_splitter.split_documents(document)

# Embed documents using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
document_texts = [doc.page_content for doc in document_chunks]
document_embeddings = embeddings.embed_documents(document_texts)
document_embeddings_array = np.array(document_embeddings)

# Build the FAISS index
d = document_embeddings_array.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
index.add(document_embeddings_array)

# Create Chroma vector database
vectordb = Chroma.from_documents(document_chunks, embedding=embeddings, persist_directory='./data')

# HuggingFace login
notebook_login()

# Load T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", device_map='auto', torch_dtype=torch.float16)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16, device_map='auto', max_length=512)
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

# Initialize Conversation Retrieval Chain
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectordb.as_retriever(search_kwargs={'k': 6})
pdf_qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, verbose=False, memory=memory)

# Streamlit interaction
st.write('Welcome to the DocBot. You are now ready to start interacting with your documents')

while True:
    query = st.text_input("Prompt:")
    if query.lower() in ["exit", "quit", "q", "f"]:
        st.write('Exiting')
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
        st.write(f"Document: {doc.page_content}")
        st.write(f"Score: {score}")
    top_document = results[0][0].page_content if results else "No relevant documents found."
    result = pdf_qa({"question": query, "context": top_document})
    st.write(f"Answer: {result['answer']}")
