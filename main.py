import os
import time
from fastapi import FastAPI, File, UploadFile
from PyPDF2 import PdfReader
from pydantic import BaseModel
from io import BytesIO
import os 
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

app = FastAPI()
DOCUMENT_EMBEDDINGS = []
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def log_query(question,answer,chunks,latency):
    log={
    "question":question,
    "answer":answer,
    "chunks":chunks,
    "latency":latency,
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

class AskRequest(BaseModel):
    question: str


@app.post("/ask")

async def ragask(request:AskRequest):
    start_time = time.time()
    if not DOCUMENT_EMBEDDINGS:
      return {"answer": "No document uploaded yet"}
    question = request.question
    top_chunks = await retrieve_chunks(question, DOCUMENT_EMBEDDINGS)
    context = ""
    for item in top_chunks:
        context += item["chunk"] + "\n"

    prompt = f"""
    Answer ONLY using the provided context. 
    context:
    {context}
    question:{question} """
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    answer = response.text
                        
    return {"answer": answer,
             "chunks": top_chunks,
              "latency": time.time() - start_time,
              "score":top_chunks[0]["score"] if top_chunks else None
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
    global DOCUMENT_EMBEDDINGS 
    DOCUMENT_EMBEDDINGS = embeddings
    print(len(DOCUMENT_EMBEDDINGS))
    print(len(DOCUMENT_EMBEDDINGS[0]["embedding"]))
    return {
            "filename":file.filename,
            "num_chunks": len(chunks),
            
            "embeddings_created": len(embeddings),
            "first_embedding_dimensions": len(embeddings[0] ["embedding"]) if embeddings else 0,
            
            }
async def generate_embeddings(chunks):
        try:
            results = []
            for chunk in chunks:
                
                response = genai.embed_content(
                    
                    model = "models/gemini-embedding-001",
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
            return []

def cosine_similarity(vec1 , vec2):
    dot_prdct = np.dot(vec1,vec2)
    magni1 = np.linalg.norm(vec1)
    magni2 = np.linalg.norm(vec2)
    if magni1 == 0 or magni2 == 0:
        return 0
    return dot_prdct / (magni1 * magni2)

async def retrieve_chunks(question,embeddings):
    try:
        store=[]
        response = genai.embed_content(
                model = "models/gemini-embedding-001",
                content = question 
            )
        question_embedding = response["embedding"]

        for item in embeddings:
            score = cosine_similarity(question_embedding, item["embedding"])
            store.append({
                "chunk": item["chunk"],
                "score": score
            })
        store = sorted(store, key=lambda x: x["score"], reverse=True)
        top_chunks = store[:3]
        return top_chunks
        
        
    except Exception as e:
        print(e)
        return []
                

  