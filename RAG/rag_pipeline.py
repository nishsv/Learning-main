import os
import numpy as np
from PyPDF2 import PdfReader
from utils import extract_text_from_pdf
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv('.env')
gpt_api_key = os.getenv('gpt_api_key')

class RAGPipeline:

    def __init__(self, data_path):
        
        self.data_path = data_path
        self.create_vecdb()

    def create_vecdb(self):
        
        data = self.parse_data()
        chunks = self.create_chunks(data)
        embeddings = self.create_embeddings(chunks)

        self.vec_db = faiss.IndexFlatL2(embeddings.shape[1])
        self.vec_db.add(np.array(embeddings))
        faiss.write_index(self.vec_db, 'vector_store.index')
        print('Vector database created successfully...')

    def parse_data(self):
        
        text = ""
        # Open the PDF file using PyPDF2
        reader = PdfReader(self.data_path)
        # Iterate through all pages and extract text
        for page in reader.pages:
            text += page.extract_text()
        return text

    def create_chunks(self, data, chunk_size=512, overlap=50):
        
        words = data.split(" ")
        self.chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            self.chunks.append(" ".join(words[i:i + chunk_size]))

        return self.chunks

    def create_embeddings(self, chunks):
        
        model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight embedding model
        embeddings = model.encode(chunks, convert_to_tensor=True)
        return embeddings

    def retrieve_related_chunks(self, query, top_k=5):
        query_embedding = self.create_embeddings(query)
        distances, indices = self.vec_db.search(np.array(query_embedding), top_k)
        return [self.chunks[i] for i in indices[0]]

    def retrive_answer(self, query):
        
        related_chunks = self.retrieve_related_chunks(query)

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

    data_path = './data/qlora_research_paper.pdf'

    rag_pipeline = RAGPipeline(data_path)

    query = ["Explain the methodology of the study."]
    answer = rag_pipeline.retrive_answer(query)

    print(answer)



if __name__ == "__main__":
    
    main()


