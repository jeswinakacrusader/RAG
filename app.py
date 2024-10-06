import streamlit as st
import PyPDF2
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import torch

# Function to load PDF
def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Initialize models and tokenizers
@st.cache_resource
def load_models():
    embedding_model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    tokenizer = AutoTokenizer.from_pretrained("chuanli11/Llama-3.2-3B-Instruct-uncensored")
    model = AutoModelForCausalLM.from_pretrained("chuanli11/Llama-3.2-3B-Instruct-uncensored")
    
    reader_llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )
    
    return embedding_model, tokenizer, reader_llm

# Streamlit app
def main():
    st.title("PDF Question Answering App")

    # Add an image
    st.image("https://cdn.leonardo.ai/users/4b94d285-249f-46ad-9304-910ad40bdcf9/generations/5e74949a-36ba-43d3-8641-1e1259567881/variations/Default_cinematic_photo_A_beautiful_queen_with_long_flowing_w_0_5e74949a-36ba-43d3-8641-1e1259567881_0.jpg?w=512", caption="PDF Q&A App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Load PDF
        pdf_text = load_pdf(uploaded_file)

        # Create knowledge base
        raw_knowledge_base = [
            LangchainDocument(page_content=pdf_text, metadata={"source": uploaded_file.name})
        ]

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Split documents
        split_docs = text_splitter.split_documents(raw_knowledge_base)

        # Load models
        embedding_model, tokenizer, reader_llm = load_models()

        # Create vector database
        knowledge_vector_database = FAISS.from_documents(
            split_docs,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE
        )

        # User input
        user_query = st.text_input("Ask a question about the PDF:")

        if user_query:
            # Retrieve relevant documents
            retrieved_docs = knowledge_vector_database.similarity_search(query=user_query, k=5)

            # Prepare context
            context = "\nExtracted PDF content:\n"
            context += "".join([f"Section {str(i+1)}:\n" + doc.page_content for i, doc in enumerate(retrieved_docs)])

            # Prepare prompt
            prompt_template = [
                {
                    "role": "system",
                    "content": "You are an AI assistant specializing in analyzing PDF documents. Using the information from the provided PDF context, give a comprehensive answer to the question. Respond only to the question asked, ensuring your response is concise and relevant. If possible, reference specific sections from the PDF. If the answer cannot be found in the PDF context, state that the information is not available in the provided documents.",
                },
                {
                    "role": "user",
                    "content": "PDF Context:\n{context}\n---\nQuestion: {question}",
                },
            ]

            rag_prompt_template = tokenizer.apply_chat_template(
                prompt_template, tokenize=False, add_generation_prompt=True
            )

            final_prompt = rag_prompt_template.format(question=user_query, context=context)

            # Generate answer
            answer = reader_llm(final_prompt)[0]["generated_text"]

            # Display answer
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
