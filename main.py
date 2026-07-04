import os
import redis
from fastapi.middleware.cors import CORSMiddleware
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
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

app = FastAPI()
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://insight-enginee-frontend.vercel.app"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

load_dotenv()

otlp_exporter = OTLPSpanExporter(
    endpoint="https://api.smith.langchain.com/otel/v1/traces",
    headers ={
        "x-api-key" : os.getenv("LANGSMITH_API_KEY"),
        "Langsmith-Project": os.getenv("LANGSMITH_PROJECT", "default")
    }
)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
tracer = trace.get_tracer(__name__)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def log_query(question,answer,chunks,latency,retrieval_distance,eval_score):
    log={ 
    "question":question,
    "answer":answer,
    "chunks":chunks,
    "latency":latency,
    "retrieval_distance":retrieval_distance,
    "eval": eval_score
    }
    try:
      with open("logs.json","r") as file:
            logs = json.load(file)
    except:
        logs = []
    
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
    with tracer.start_as_current_span("ask_route") as span:
        start_time = time.time()
        if collection.count()==0:
         return {"answer": "No document uploaded yet"}
        question = request.question
        cache_key = f"question:{question}"

    #     cached_answer = redis_client.get(cache_key)

    #     if cached_answer:
    #        return {
    #     "answer": cached_answer,
    #     "cached": True
    # }
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
        with tracer.start_as_current_span("generate_answer") as llm_span:
         llm_span.set_attribute("langsmith.span.kind", "LLM")
         llm_span.set_attribute("gen_ai.system", "Gemini")
         llm_span.set_attribute("gen_ai.request.model", "gemini-2.5-flash")
         llm_span.set_attribute("gen_ai.prompt.0.content", prompt)
         

         model = genai.GenerativeModel("gemini-2.5-flash")
         response = model.generate_content(prompt)
         answer = response.text
        #  redis_client.setex(
        #      cache_key,
        #       3600,
        #      answer
        #     )
         llm_span.set_attribute("gen_ai.completion.0.content", answer)
    

        evaluation_score = await evaluate_answer(answer, top_chunks)
        latency = time.time() - start_time
        
        span.set_attribute("question", question)
        span.set_attribute("retrieved_chunks", len(top_chunks))
        span.set_attribute("retrieval_distance", top_chunks[0]["distance"])
        span.set_attribute("latency", latency)
        span.set_attribute("grounded", evaluation_score["grounded"])
        span.set_attribute("relevance_score", evaluation_score["relevance_score"])
        log_query(question,answer,top_chunks,latency,top_chunks[0]["distance"] if top_chunks else None,evaluation_score)                 
        return {"answer": answer,
             "chunks": top_chunks,
              "latency": latency,
              "retrieval_distance":top_chunks[0]["distance"] if top_chunks else None,
              "eval": evaluation_score,
              "cached": False,
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
      try:
        text = content.decode("utf-8")
      except:
        text = content.decode("cp1252")
      text = text.replace("\xa0", " ")
      text = " ".join(text.split())
        
        
    elif file.filename.endswith("pdf"):
        
        pdf_file = BytesIO(content)
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text+=page_text + "\n"
        text = text.replace("\\n", " ")
        text = text.replace("\n", " ")
        text = " ".join(text.split())

    
        
    if not text.strip():
        return {"error":"no text found in the document"}

    chunks = chunk_text(text)
    if not chunks:
        return {"error":"no text found in the file"}
    embeddings = await generate_embeddings(chunks)
    ids=[]
    documents=[]
    vectors=[]
    for i,item in enumerate(embeddings):
        ids.append(f"{file.filename}_chunk_{i}")
        documents.append(item["chunk"])
        vectors.append(item["embedding"])
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
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
        with tracer.start_as_current_span("retrieve_chunks") as db_span:
          db_span.set_attribute("langsmith.span.kind", "retriever")
          db_span.set_attribute("db.system", "chromadb")
          db_span.set_attribute("input.value", question) 
          

        
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
                "distance": results["distances"][0][i],
            })
          db_span.set_attribute("output.value", str(top_chunks)) 
          return top_chunks
        
    except Exception as e:
        print(e)
        return []

async def evaluate_answer(answer,top_chunks):
    context = ""
    for item in top_chunks:
        context += item["chunk"] + "\n"
    match_words= 0;
    for word in answer.split():
        if word in context:
            match_words += 1

    grounded = match_words > 3
    relevance_score = 0
    if grounded:
        relevance_score = 1.0
    else:
        relevance_score = 0.5
    return {
        "grounded":grounded,
        "relevance_score":relevance_score
    }


  