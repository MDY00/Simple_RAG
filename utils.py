from langchain_chroma import Chroma
from langchain.schema.document import Document
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata["source"]
        page = chunk.metadata["page"]
        current_page_id = f"{source}_{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks , DB_PATH, hf_embeddings):
    db = Chroma(
        persist_directory=DB_PATH, embedding_function=hf_embeddings
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = existing_items["ids"]
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    chunks_to_add = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            chunks_to_add.append(chunk)

    if len(chunks_to_add):
        print(f"Adding new documents: {len(chunks_to_add)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in chunks_to_add]
        db.add_documents(chunks_to_add, ids=new_chunk_ids)
    else:
        print("No new documents to add")
    