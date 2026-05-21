import os
import time
from fastapi import FastAPI, File, UploadFile
from PyPDF2 import PdfReader
from pydantic import BaseModel
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import json
import chromadb

app = FastAPI()
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

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
    with open("logs.json","r") as file:
        logs = json.load(file)
    logs.append(log)
    with open("logs.json","w") as file:
        json.dump(logs,file,indent=4)
@app.get("/health")
def health_check():
    return {"status": "ok"}

class AskRequest(BaseModel):
    question: str


@app.post("/ask")

async def ragask(request:AskRequest):
    start_time = time.time()
    if collection.count()==0:
      return {"answer": "No document uploaded yet"}
    question = request.question
    top_chunks = await retrieve_chunks(question)
    if not top_chunks:
        return {"answer": "No relevant information found in the document"}
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
    latency = time.time() - start_time
    log_query(question,answer,top_chunks,latency)                 
    return {"answer": answer,
             "chunks": top_chunks,
              "latency": latency,
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
    if not text.strip():
        return {"error":"no text found in the document"}

    chunks = chunk_text(text)
    embeddings = await generate_embeddings(chunks)
    ids=[]
    documents=[]
    vectors=[]
    for i,item in enumerate(embeddings):
        ids.append(f"{file.filename}_chunk_{i}")
        documents.append(item["chunk"])
        vectors.append(item["embedding"])
    collection.delete(
        ids= ids
    ) 
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=vectors
    )
   

    
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

async def retrieve_chunks(question):
    try:
        
        response = genai.embed_content(
                model = "models/gemini-embedding-001",
                content = question 
            )
        question_embedding = response["embedding"]

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )

        top_chunks = []
        for i in range(len(results["documents"][0])):
            top_chunks.append({
                "chunk": results["documents"][0][i],
                "score": results["distances"][0][i],
            })
        return top_chunks
        
    except Exception as e:
        print(e)
        return []

async def

  