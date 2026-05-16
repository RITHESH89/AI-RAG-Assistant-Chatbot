import os
import streamlit as st
from utils.pdf_loader import load_pdf
from utils.embeddings import create_vectorstore, load_vectorstore
from utils.rag_chain import get_rag_chain

os.makedirs("data", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

st.set_page_config(page_title="AI RAG Assistant")

st.title("AI RAG Assistant Chatbot")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    pdf_path = f"data/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully")

    with st.spinner("Processing PDF..."):

        chunks = load_pdf(pdf_path)

        create_vectorstore(chunks)

    st.success("Vector Database Created")

question = st.text_input("Ask Question From PDF")

if question:

    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(question, k=3)

    chain = get_rag_chain()

    response = chain.invoke({
        "input_documents": docs,
        "question": question
    })

    st.subheader("Answer")
    st.write(response["output_text"])
