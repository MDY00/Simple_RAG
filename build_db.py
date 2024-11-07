import os
import shutil
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from utils import add_to_chroma

load_dotenv() 

DATA_PATH = os.getenv("DATA_PATH")

DB_PATH = os.getenv("DB_PATH")
MODEL_NAME = os.getenv("MODEL_NAME")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE") == 'True'

hf_embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'trust_remote_code': TRUST_REMOTE_CODE}
)

# if os.path.exists(DB_PATH):
#     shutil.rmtree(DB_PATH)

documents = PyPDFDirectoryLoader(DATA_PATH).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(chunks)} chunks")

add_to_chroma(chunks, DB_PATH, hf_embeddings)
