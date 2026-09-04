"""
Agri-Mitra AI Chatbot & Agronomic Co-Pilot — Backend API Service
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers.copilot import copilot_router, ocr_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agrimitra-chatbot")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & lifecycle management."""
    logger.info("🌱 Starting Agri-Mitra AI Chatbot & Agronomic Co-Pilot Backend API...")
    logger.info(f"RAG Docs Directory: {settings.RAG_DOCS_DIR}")
    yield
    logger.info("Agri-Mitra AI Chatbot service stopped.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Multilingual RAG Agronomic Co-Pilot API with Speech Synthesis & Land Document Intelligence",
    version=settings.VERSION,
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(copilot_router, prefix="/api/v1")
app.include_router(ocr_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "query": "/api/v1/copilot/query",
            "quick_prompts": "/api/v1/copilot/quick-prompts",
            "tts": "/api/v1/copilot/tts",
            "tts_status": "/api/v1/copilot/tts/status",
            "ocr": "/api/v1/ocr/pahani",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
