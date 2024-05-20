import streamlit as st
import os
import sys
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import notebook_login

st.title("Document-based Chatbot")

uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["pdf", "docx", "txt"])

if uploaded_files:
    document = []
    for uploaded_file in uploaded_files:
        st.write("Uploaded file:", uploaded_file)  # Debugging statement
        st.write("File name:", uploaded_file.name)  # Debugging statement
        st.write("File type:", uploaded_file.type)  # Debugging statement
        st.write("File size:", uploaded_file.size)  # Debugging statement
        if hasattr(uploaded_file, 'content_type'):
            st.write("File content type:", uploaded_file.content_type)  # Debugging statement
        st.write("File attributes:", dir(uploaded_file))  # Debugging statement
        if uploaded_file.type == "application/pdf":
            st.write("Processing PDF file...")  # Debugging statement
            loader = PyPDFLoader(uploaded_file)
            document.extend(loader.load())
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            st.write("Processing DOCX file...")  # Debugging statement
            loader = Docx2txtLoader(uploaded_file)
            document.extend(loader.load())
        elif uploaded_file.type == "text/plain":
            st.write("Processing TXT file...")  # Debugging statement
            loader = TextLoader(uploaded_file)
            document.extend(loader.load())

    document_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
    document_chunks = document_splitter.split_documents(document)

    # Embed documents and build FAISS index
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    document_texts = [doc.page_content for doc in document_chunks]
    document_embeddings = embeddings.embed_documents(document_texts)
    document_embeddings_array = np.array(document_embeddings)

    # Build the FAISS index
    d = document_embeddings_array.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
    index.add(document_embeddings_array)

    vectordb = Chroma.from_documents(document_chunks, embedding=embeddings, persist_directory='./data')
    vectordb.persist()

    # Authentication for HuggingFace
    notebook_login()

    # Load the FLAN-T5 model and tokenizer
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

    st.write('---')
    st.write("Welcome to the DocBot. You are now ready to start interacting with your documents.")

    query = st.text_input("Enter your query:", "")

    if query:
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
else:
    st.write("Please upload documents to proceed.")
