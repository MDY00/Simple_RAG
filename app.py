import os
import shutil
import streamlit as st
from transformers import pipeline
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from utils import add_to_chroma

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
DB_PATH = os.getenv("DB_PATH")
MODEL_NAME = os.getenv("MODEL_NAME")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE") == 'True'
DEVICE = int(os.getenv("DEVICE"))

hf_embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'trust_remote_code': TRUST_REMOTE_CODE}
)

db = Chroma(persist_directory=DB_PATH, embedding_function=hf_embeddings)

template = """
Answer the question based on the context below. If you can't answer the question, write "I don't know".

Context: {context}

Question: {question}
"""

st.set_page_config(page_title="RAG System", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) System")

st.sidebar.title("Menu")
option = st.sidebar.selectbox("Choose Action", ["Add PDF Documents", "Ask a Question"])

if option == "Add PDF Documents":
    st.header("Add New PDF Documents to the Database")

    uploaded_files = st.file_uploader("Select PDF files", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        documents = [PyPDFLoader(os.path.join(DATA_PATH, file.name)).load() for file in uploaded_files]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        chunks = text_splitter.split_documents(sum(documents, []))
        add_to_chroma(chunks, DB_PATH, hf_embeddings)
        
        st.success(f"Added {len(uploaded_files)} files to the database!")

elif option == "Ask a Question":
    st.header("Ask a Question")

    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question:
            result = db.similarity_search(question, k=2)
            context_text = "\n\n".join([chunk.page_content for chunk in result])

            prompt_template = PromptTemplate.from_template(template)
            prompt = prompt_template.format(context=context_text, question=question)

            pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=DEVICE)
            answer = pipe(prompt, max_length=50)
            sources_list = [chunk.metadata["id"] for chunk in result]

            st.write(f"**Answer:** {answer[0]['generated_text']}")
            st.write("**Sources:**")
            for source in sources_list:
                st.write(f"- {source}")
