import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

vectordb = Chroma(persist_directory='./data')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

st.title("DocBot")
st.write("Welcome to the DocBot. You are now ready to start interacting with your knowledge base.")

query = st.text_input("Enter your query:")
top_n = st.number_input("Number of top results to consider:", min_value=1, value=3, step=1)

if st.button("Get Answer"):
    if query:
        # Compute the query embedding
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = embeddings.embed_documents([query])[0]

        results = vectordb.similarity_search_with_score(query_embedding, k=top_n)

        st.write("Top relevant document chunks:")
        for doc, score in results:
            st.write(f"Document: {doc.page_content}")
            st.write(f"Score: {score}")

        top_documents = " ".join([doc.page_content for doc, _ in results])
        result = pipe(f"Based on the following documents: {top_documents}\nAnswer the question: {query}")

        # Display the generated answer
        st.write("Answer:")
        st.write(result[0]['generated_text'])
    else:
        st.write("Please enter a query.")
