"""
Agri-Mitra Chatbot — Application Configuration
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "Agri-Mitra AI Chatbot & Agronomic Co-Pilot"
    VERSION: str = "2.0.0"

    # External LLM & RAG APIs
    GOOGLE_API_KEY: str = ""
    
    # ElevenLabs Voice Synthesis & Regional Audio
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"
    ELEVENLABS_MODEL_ID: str = "eleven_multilingual_v2"

    # OCR Path for Pahani/RTC Land Documents
    TESSERACT_CMD: str = "C:/Program Files/Tesseract-OCR/tesseract.exe"

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:8000,*"

    # RAG Storage & Document Paths
    CHROMA_PERSIST_DIR: str = os.path.abspath("./chroma_db")
    RAG_DOCS_DIR: str = os.path.abspath("./rag_docs")

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
