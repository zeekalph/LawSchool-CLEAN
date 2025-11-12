from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from RagFullPipeline import (
        process_all_pdfs,
        split_documents,
        EmbeddingManager,
        VectorStore,
        RagRetriever,
        rag_advanced,
        ChatGroq
    )

    RAG_AVAILABLE = True
    print("✅ Successfully imported RagFullPipeline")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative import...")
    try:
        from app.RagFullPipeline import (
            process_all_pdfs,
            split_documents,
            EmbeddingManager,
            VectorStore,
            RagRetriever,
            rag_advanced,
            ChatGroq
        )

        RAG_AVAILABLE = True
        print("✅ Successfully imported from app.RagFullPipeline")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        RAG_AVAILABLE = False


        # Create dummy functions for development
        def process_all_pdfs(*args, **kwargs):
            return []


        def split_documents(*args, **kwargs):
            return []


        class EmbeddingManager:
            def __init__(self, *args, **kwargs): pass

            def generate_embeddings(self, *args, **kwargs): return []


        class VectorStore:
            def __init__(self, *args, **kwargs):
                self.collection = type('obj', (object,), {'count': lambda: 0})()


        class RagRetriever:
            def __init__(self, *args, **kwargs): pass

            def retrieve(self, *args, **kwargs): return []


        def rag_advanced(*args, **kwargs):
            return {'answer': 'RAG system not available', 'sources': [], 'confidence': 0.0}


        class ChatGroq:
            def __init__(self, *args, **kwargs): pass


class Settings:
    API_TITLE = "Law Study Buddy - Advanced RAG API"
    API_DESCRIPTION = "AI-powered legal research assistant using RAG technology"
    API_VERSION = "1.0.0"

    # File paths
    PDF_DIRECTORY = "C:/Users/DELL/LawSchool0.2/pdf_files"
    VECTOR_STORE_DIR = "C:/Users/DELL/LawSchool0.2/vector_store"

    # RAG Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    COLLECTION_NAME = "pdf_documents"

    # Retrieval Configuration
    DEFAULT_TOP_K = 5
    DEFAULT_MIN_SCORE = 0.2

    # LLM Configuration - Get from environment variables
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # FIXED: Use environment variable
    LLM_MODEL = "llama-3.1-8b-instant"
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 3035


settings = Settings()


# Pydantic Models (keep the same)
class QueryRequest(BaseModel):
    query: str = Field(..., description="The legal question to answer")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    min_score: Optional[float] = Field(0.0, description="Minimum similarity score threshold")
    return_context: Optional[bool] = Field(False, description="Whether to return full context")


class SourceDocument(BaseModel):
    source: str = Field(..., description="Document source file")
    page: str = Field(..., description="Page number if available")
    score: float = Field(..., description="Similarity score")
    preview: str = Field(..., description="Content preview")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(..., description="Source documents used")
    confidence: float = Field(..., description="Overall confidence score")
    context: Optional[str] = Field(None, description="Full context if requested")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    vector_store_count: Optional[int] = Field(None, description="Number of documents in vector store")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    rag_available: bool = Field(..., description="Whether RAG system is available")


class InitializeResponse(BaseModel):
    message: str = Field(..., description="Initialization status message")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error description")
    error_type: str = Field(..., description="Type of error")


# Global instances
rag_components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    print("🚀 Initializing Law Study Buddy RAG API...")

    if RAG_AVAILABLE:
        try:
            # Initialize LLM with environment variable
            llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,  # Now uses environment variable
                model_name=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            )

            # Initialize embedding manager
            embedding_manager = EmbeddingManager()

            # Initialize vector store
            vectorstore = VectorStore(
                collection_name=settings.COLLECTION_NAME,
                persist_directory=settings.VECTOR_STORE_DIR
            )

            # Initialize retriever
            rag_retriever = RagRetriever(vectorstore, embedding_manager)

            # Store components globally
            rag_components.update({
                'llm': llm,
                'embedding_manager': embedding_manager,
                'vectorstore': vectorstore,
                'rag_retriever': rag_retriever,
                'available': True
            })

            print("✅ RAG pipeline initialized successfully")
            print(f"📊 Vector store documents: {vectorstore.collection.count()}")

        except Exception as e:
            print(f"❌ RAG pipeline initialization failed: {e}")
            rag_components.update({'available': False})
    else:
        print("❌ RAG components not available - running in limited mode")
        rag_components.update({'available': False})

    yield  # Application runs here

    # Shutdown: Cleanup resources
    print("🛑 Shutting down RAG pipeline...")
    rag_components.clear()


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get RAG components
def get_rag_components():
    """Dependency that provides RAG components to endpoints"""
    if not rag_components.get('available', False):
        raise HTTPException(
            status_code=503,
            detail="RAG components not available. Please check if RagFullPipeline.py is properly set up."
        )
    return rag_components


# Exception handlers (keep the same)
@app.exception_handler(500)
async def internal_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error",
            error_type="InternalError"
        ).dict()
    )


@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            detail="Validation error - check your request parameters",
            error_type="ValidationError"
        ).dict()
    )


@app.exception_handler(503)
async def service_unavailable_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            detail="Service temporarily unavailable - RAG components not initialized",
            error_type="ServiceUnavailable"
        ).dict()
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check with system status"""
    try:
        if rag_components.get('available', False):
            vector_count = rag_components['vectorstore'].collection.count()
            return HealthResponse(
                status="healthy",
                vector_store_count=vector_count,
                model_loaded=True,
                rag_available=True
            )
        else:
            return HealthResponse(
                status="degraded",
                vector_store_count=0,
                model_loaded=False,
                rag_available=False
            )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            vector_store_count=0,
            model_loaded=False,
            rag_available=False
        )


# Main query endpoint
@app.post("/ask", response_model=QueryResponse, tags=["RAG Operations"])
async def ask_question(
        request: QueryRequest,
        components: dict = Depends(get_rag_components)
):
    """
    Ask a legal question and get an AI-generated answer with sources.
    """
    try:
        # Validate input parameters
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty. Please provide a legal question."
            )

        if request.top_k and (request.top_k > 20 or request.top_k < 1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_k must be between 1 and 20"
            )

        if request.min_score and (request.min_score < 0.0 or request.min_score > 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="min_score must be between 0.0 and 1.0"
            )

        print(f"🔍 Processing query: '{request.query}'")

        # Process the query using the RAG pipeline
        result = rag_advanced(
            query=request.query,
            retriever=components['rag_retriever'],
            llm=components['llm'],
            top_k=request.top_k or settings.DEFAULT_TOP_K,
            min_score=request.min_score or settings.DEFAULT_MIN_SCORE,
            return_context=request.return_context or False
        )

        print(f"✅ Query processed successfully. Confidence: {result.get('confidence', 0.0):.2f}")

        return QueryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your query. Please try again."
        )




if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )