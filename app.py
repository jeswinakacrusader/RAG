import streamlit as st
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import matplotlib.pyplot as plt
import pacmap
import numpy as np
import plotly.express as px

# Streamlit page configuration
st.set_page_config(page_title="RAG Model App", layout="wide")

# Load necessary models and vectorstore
@st.cache_resource(show_spinner=False)
def load_resources():
    # Load embedding model and knowledge base
    embedding_model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load FAISS vector store
    knowledge_vector_database = FAISS.from_documents(
        docs_processed, embedding_model
    )
    
    # Load reader model
    reader_model_name = "HuggingFaceH4/zephyr-7b-beta"
    model = AutoModelForCausalLM.from_pretrained(reader_model_name)
    tokenizer = AutoTokenizer.from_pretrained(reader_model_name)
    reader_llm = pipeline(model=model, tokenizer=tokenizer, task="text-generation")

    return embedding_model, knowledge_vector_database, reader_llm

# Load models and data
embedding_model, knowledge_vector_database, reader_llm = load_resources()

st.title("Retrieval-Augmented Generation (RAG) Demo")

# Input for user query
user_query = st.text_input("Enter your question:", "How to create a pipeline object?")

if user_query:
    st.write("Retrieving documents...")
    query_vector = embedding_model.embed_query(user_query)
    
    # Retrieve similar documents
    retrieved_docs = knowledge_vector_database.similarity_search(query=user_query, k=5)
    
    # Show retrieved documents
    for i, doc in enumerate(retrieved_docs):
        st.write(f"**Document {i+1}:**")
        st.write(doc.page_content[:500] + "...")
    
    # Show the final generated response from the LLM
    st.write("Generating answer...")
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n" + "".join([f"Document {i}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

    # Construct prompt and generate answer
    prompt = f"Context:\n{context}\nQuestion: {user_query}"
    answer = reader_llm(prompt)[0]["generated_text"]
    st.write(f"**Answer:** {answer}")

    # Optional: Visualize embeddings using PaCMAP
    if st.checkbox("Visualize Embeddings"):
        st.write("Visualizing document embeddings...")
        embedding_projector = pacmap.PaCMAP(n_components=2)
        embeddings_2d = embedding_projector.fit_transform(np.array([embedding_model.embed_documents([doc]) for doc in retrieved_docs]))
        df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
        df['text'] = [doc.page_content[:100] for doc in retrieved_docs]
        fig = px.scatter(df, x='x', y='y', hover_data=['text'])
        st.plotly_chart(fig)
