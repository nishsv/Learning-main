import os
import numpy as np
from utils import extract_text_from_pdf
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv('.env')
gpt_api_key = os.getenv('gpt_api_key')


def extract_text_from_pdf(pdf_path):
    text = ""
    # Open the PDF file using PyPDF2
    reader = PdfReader(pdf_path)
    # Iterate through all pages and extract text
    for page in reader.pages:
        text += page.extract_text()
    return text

def retrieve(query, index, chunks, top_k=5):
    query_embedding = generate_embeddings(query)
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def create_vector_db(embeddings):

    vec_db = faiss.IndexFlatL2(embeddings.shape[1])
    vec_db.add(np.array(embeddings))
    faiss.write_index(vec_db, 'vector_store.index')

    return vec_db

def generate_embeddings(chunks):
    
    model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight embedding model
    embeddings = model.encode(chunks, convert_to_tensor=True)
    
    return embeddings

def split_text(text, chunk_size=512, overlap=50):
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def generate_response_llm(gpt_api_key, query, related_chunks):

    client = OpenAI(api_key=gpt_api_key)

    response = client.chat.completions.create(

        messages=[
            {
                'role' : 'system',
                'content' : ("".join(related_chunks))
            },
            {
                'role' : 'user',
                'content' : ("".join(query))
            }
        ],
        model='gpt-4',
        temperature=0.01,
        max_tokens=4096,
        top_p=1
    )

    resp = response.choices[0].message.content

    return resp

def main():

    research_paper_path = './data/qlora_research_paper.pdf'

    # Step 1 : Extract the content from documents
    data = extract_text_from_pdf(research_paper_path)
   
    # Step 2 : Split the data into multiple chunks
    chunks = split_text(data)

    # Step 3 : Convert the chunks into vector embeddings
    embeddings = generate_embeddings(chunks)

    # Step 4 : Create a vector DB and store the embeddings
    vector_db = create_vector_db(embeddings)

    # Step 5 : Query from User and retrive the related chunks
    query = ["Explain the methodology of the study."]
    relevant_chunks = retrieve(query, vector_db,chunks)

    # Step 6 : Initialize LLM (GPT)
    answer = generate_response_llm(gpt_api_key, query, relevant_chunks)

    print(query)
    print(answer)



if __name__ == '__main__':
    main()