import argparse
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

load_dotenv() 

DB_PATH = os.getenv("DB_PATH")
MODEL_NAME = os.getenv("MODEL_NAME")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE") == 'True'
DEVICE = int(os.getenv("DEVICE"))

hf_embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'trust_remote_code': TRUST_REMOTE_CODE}
)

template = """
Answer the question based on the context below. If you can't answer the question, write "I don't know".

Context: {context}

Question: {question}
"""

def main(question):
    db = Chroma(persist_directory=DB_PATH, embedding_function=hf_embeddings)
    result = db.similarity_search(question, k=2)

    context_text = "\n\n".join([chunk.page_content for chunk in result])

    prompt_template = PromptTemplate.from_template(template)
    prompt = prompt_template.format(context=context_text, question=question)

    pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=DEVICE)
    answer = pipe(prompt, max_length=50)
    sources_list = [chunk.metadata["id"] for chunk in result]

    print(f"Answer: {answer[0]['generated_text']}")
    print("Sources:\n" + "\n".join(f"- {source}" for source in sources_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a question.")
    parser.add_argument("question", type=str, help="The question to be answered")
    args = parser.parse_args()
    main(args.question)
