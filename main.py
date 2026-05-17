import os

from fastapi import FastAPI, File, UploadFile
from PyPDF2 import PdfReader
from pydantic import BaseModel
from io import BytesIO
import os 
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

app = FastAPI()

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
@app.get("/health")
def health_check():
    return {"status": "ok"}

class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ragask(request:AskRequest):
    return {"answer": f"you asked: {request.question}",
             "chunks":[],
              "latency":None,
              "score":None
            }

def chunk_text(text):
    chunk_size = 500
    overlap = 100
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    return chunks
@app.post("/upload")
async def upload_file(file:UploadFile = File(...)):
    content = await file.read()
    text = ""
    
    if file.filename.endswith("txt"):
        text = content.decode("utf-8")
        
    elif file.filename.endswith("pdf"):
        
        pdf_file = BytesIO(content)
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text+=page_text

    chunks = chunk_text(text)
    embeddings = await generate_embeddings(chunks)
    return {
            "filename":file.filename,
            "num_chunks": len(chunks),
            "chunks" : chunks,
            "embeddings_created": len(embeddings),
            "first_embedding_dimensions": len(embeddings[0] ["embedding"]),
            
            }
async def generate_embeddings(chunks):
        try:
            results = []
            for chunk in chunks:
                
                response = genai.embed_content(
                    
                    model = "gemini-embedding-2",
                    content = chunk
                )
                
                embedding = response["embedding"]

            
                results.append({
                "chunk": chunk,
                "embedding": embedding
            })
            return results
        except Exception as e:
            print(e)

def cosing_similarity(vec1 , vec2):
    dot_prdct = np.dot(vec1,vec2)
    magni1 = np.linalg.norm(vec1)
    magni2 = np.linalg.norm(vec2)
    solvec = dot_prdct / (magni1 * magni2)
    return solvec

async def retrieve_chunks(question,embeddings):
    try:
        store=[]
        response = genai.embed_content(
                model = "gemini-embedding-2",
                content = question 
            )
        question_embedding = response["embedding"]

        for item in embeddings:
            score = cosing_similarity(question_embedding, item["embedding"])
            store.append({
                "chunk": item["chunk"],
                "score": score
            })
        return store
    except Exception as e:
        print(e)
                

  