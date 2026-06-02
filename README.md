Insight Engine

Production-ready AI-powered RAG platform that allows users to upload documents and ask contextual questions using LLMs. Built with FastAPI, ChromaDB, Gemini, React, and modern AI observability tooling.

Live Demo


https://insight-enginee-frontend.vercel.app/



Features
Upload and index documents
Semantic search using vector embeddings
Context-aware question answering (RAG pipeline)
Real-time AI responses with source retrieval
AI evaluation metrics:
Groundedness
Relevance score
Retrieval distance
Latency tracking
LangSmith tracing + observability
Production-ready FastAPI backend
Modern React + TypeScript frontend
Dockerized architecture
Redis-ready caching architecture
Fully deployed frontend and backend
Tech Stack
Backend
Python
FastAPI
ChromaDB
LangChain
Gemini API
LangSmith
OpenTelemetry
Docker
Redis (planned integration)
Frontend
React
TypeScript
Vite
Axios
TailwindCSS
Deployment
Render (Backend)
Vercel (Frontend)


Key Engineering Highlights - :
Built scalable async FastAPI APIs for document ingestion and querying
Implemented vector similarity search with ChromaDB
Added retrieval evaluation metrics and latency monitoring
Integrated LangSmith tracing for observability and debugging
Developed production-style RAG workflow with semantic retrieval
Built responsive AI frontend with real-time UX feedback
Configured CORS, deployment pipelines, and cloud hosting
Structured codebase for production scalability and modularity
API Endpoints
Upload Document
POST /upload

Uploads and indexes a document into the vector database.

Ask Question
POST /ask

Example Request:

{
  "question": "What backend skills are mentioned?"
}

Example Response:

{
  "answer": "Python, FastAPI",
  "latency": 1.34,
  "retrieval_distance": 0.64,
  "eval": {
    "grounded": true,
    "relevance_score": 1.0
  }
}
Local Setup
Backend
git clone <repo-url>
cd insight-engine-backend

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload
Frontend
cd frontend

npm install

npm run dev
Environment Variables
GEMINI_API_KEY=your_key
LANGSMITH_API_KEY=your_key
Upcoming Improvements
Redis response caching
Docker Compose setup
Hybrid search
Reranking pipeline
Multi-document retrieval
Agentic AI workflows using LangGraph
Authentication system
Streaming responses
File type expansion (PDF/DOCX)



What I Learned -:

Building production-ready RAG systems
AI observability and evaluation
Vector databases and embeddings
Full-stack AI engineering
Deployment and cloud infrastructure
LLM application architecture
Retrieval optimization
Modern AI backend engineering
