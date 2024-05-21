import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Sample document chunks to form a knowledge base
document_chunks = [
    "Reachability analysis (RA) has been widely employed to formally verify driving safety.",
    "RA computes a complete set of states that an agent can reach given an initial condition.",
    "Safety verification can be performed by propagating all possible reachable spaces.",
    "The Forward Reachable Set (FRS) of the automated vehicle does not intersect with other participants.",
    "Prediction-based confidence-aware stochastic forward reachable set (FRS) is calculated online.",
    # Add more chunks as needed
]

# Initialize embeddings model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute embeddings for the document chunks
document_embeddings = np.array([embedding_model.encode(doc) for doc in document_chunks])

# Initialize FAISS index
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)

# Initialize the LLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Define Streamlit app
st.title("DocBot")
st.write("Welcome to the DocBot. You are now ready to start interacting with your knowledge base.")

query = st.text_input("Enter your query:")
top_n = st.number_input("Number of top results to consider:", min_value=1, value=3, step=1)

if st.button("Get Answer"):
    if query:
        # Compute the query embedding
        query_embedding = np.array(embedding_model.encode(query)).reshape(1, -1)
        
        # Perform FAISS similarity search
        distances, indices = index.search(query_embedding, top_n)
        
        # Retrieve the top N document chunks
        results = [(document_chunks[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        
        # Display the top N document chunks and their scores
        st.write("Top relevant document chunks:")
        for doc, score in results:
            st.write(f"Document: {doc}")
            st.write(f"Score: {score}")
        
        # Concatenate the top documents to form the context
        top_documents = " ".join([doc for doc, _ in results])
        
        # Generate a response using the LLM
        result = pipe(f"Based on the following documents: {top_documents}\nAnswer the question: {query}")
        
        # Display the generated answer
        st.write("Answer:")
        st.write(result[0]['generated_text'])
    else:
        st.write("Please enter a query.")
