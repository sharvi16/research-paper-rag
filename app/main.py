from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.ingestion.embedder import PaperEmbedder
from app.retrieval.retriever import PaperRetriever
from app.generation.generator import RAGGenerator
from app.api.routes import router

# Global dict to store heavy ML components across requests
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan hook to safely load models and connect to databases 
    upon server startup, and clean up upon shutdown.
    """
    print("Initializing ML Services...")
    
    try:
        # Load SentenceTransformers and connect to ChromaDB
        embedder = PaperEmbedder.from_existing() 
        retriever = PaperRetriever(embedder)
        
        # Initialize LangChain and ChatGroq
        generator = RAGGenerator()
        
        services["embedder"] = embedder
        services["retriever"] = retriever
        services["generator"] = generator
        
        print("ML Services initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR during startup: {e}")
        # Depending on setup, you might want to suppress or re-raise
        
    yield
    
    # Shutdown sequence
    print("Shutting down services...")
    services.clear()


app = FastAPI(title="ML Paper RAG API", lifespan=lifespan)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to safely inject the global ML services into the Request State
# This allows routers to cleanly access the embedder/generator without global variables.
@app.middleware("http")
async def inject_services(request, call_next):
    request.state.embedder = services.get("embedder")
    request.state.retriever = services.get("retriever")
    request.state.generator = services.get("generator")
    response = await call_next(request)
    return response

# Register our API routes
app.include_router(router)

# Mount the static directory to serve the frontend interface.
# Since it's mounted at "/", any request that doesn't match an API route 
# will fall back to serving static files (e.g. index.html)
import os
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
