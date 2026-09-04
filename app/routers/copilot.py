"""
Agri-Mitra — RAG Co-Pilot & OCR Routers
Provides high-performance agricultural RAG query, regional voice synthesis, and land record OCR.
"""
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.services.rag_service import rag_service
from app.services.ocr_service import parse_pahani

# Co-Pilot router
copilot_router = APIRouter(prefix="/copilot", tags=["AI Co-Pilot"])


class CoPilotQuery(BaseModel):
    question: str
    language: str = "English"
    field_id: str | None = None
    field_context: dict | None = None


@copilot_router.post("/query")
async def query_copilot(req: CoPilotQuery):
    """
    Query the RAG agronomic co-pilot.
    Optionally enrich with field telemetry context.
    """
    result = rag_service.query(
        question=req.question,
        language=req.language,
        field_context=req.field_context,
    )
    return result


@copilot_router.get("/quick-prompts")
async def quick_prompts(language: str = "English"):
    """Return localized quick prompt suggestions for the UI."""
    prompts = {
        "English": [
            "What is my yield forecast?",
            "Is my soil pH optimal for rice?",
            "How much nitrogen should I apply?",
            "When should I irrigate?",
            "What crop is best for my soil?",
        ],
        "Hindi": [
            "मेरी फसल का उत्पादन पूर्वानुमान क्या है?",
            "क्या मेरी मिट्टी का pH चावल के लिए सही है?",
            "मुझे कितना नाइट्रोजन देना चाहिए?",
            "कब सिंचाई करनी चाहिए?",
        ],
        "Kannada": [
            "ನನ್ನ ಇಳುವರಿ ಮುನ್ಸೂಚನೆ ಏನು?",
            "ನನ್ನ ಮಣ್ಣಿನ pH ಭತ್ತಕ್ಕೆ ಸರಿಯಾಗಿದೆಯೇ?",
            "ನಾನು ಎಷ್ಟು ಸಾರಜನಕ ಹಾಕಬೇಕು?",
        ],
        "Telugu": [
            "నా దిగుబడి అంచనా ఎంత?",
            "వరికి నా నేల pH సరిగ్గా ఉందా?",
            "నేను ఎంత నత్రజని వేయాలి?",
            "ఎప్పుడు నీరు పెట్టాలి?",
        ],
        "Tamil": [
            "எனது விளைச்சல் முன்னறிவிப்பு என்ன?",
            "நெல் பயிருக்கு என் மண் pH உகந்ததா?",
            "நான் எவ்வளவு நைட்ரஜன் இட வேண்டும்?",
        ],
    }
    return {"prompts": prompts.get(language, prompts["English"])}


class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    voice_id: str | None = None
    custom_api_key: str | None = None


@copilot_router.post("/tts")
async def synthesize_speech(req: TTSRequest):
    """
    Synthesize regional speech using ElevenLabs Multilingual V2 or neural stream fallback.
    Returns audio/mpeg stream.
    """
    from fastapi import Response
    from app.services.elevenlabs_service import elevenlabs_service

    audio_bytes, error = await elevenlabs_service.synthesize_speech(
        text=req.text,
        voice_id=req.voice_id,
        language=req.language,
        custom_api_key=req.custom_api_key,
    )

    if error or not audio_bytes:
        return {"available": False, "reason": error or "Failed to synthesize speech"}

    return Response(content=audio_bytes, media_type="audio/mpeg")


@copilot_router.get("/tts/status")
async def tts_status():
    """Returns whether ElevenLabs Regional Voice is configured."""
    from app.services.elevenlabs_service import elevenlabs_service
    from app.core.config import get_settings
    settings = get_settings()
    return {
        "elevenlabs_active": elevenlabs_service.is_configured(),
        "model": settings.ELEVENLABS_MODEL_ID,
        "voice_id": settings.ELEVENLABS_VOICE_ID,
    }


# OCR Router
ocr_router = APIRouter(prefix="/ocr", tags=["OCR & Land Records"])


@ocr_router.post("/pahani")
async def upload_pahani(file: UploadFile = File(...)):
    """
    Upload Pahani/RTC/7-12 Satbara PDF for OCR extraction.
    Returns parsed data for human review.
    """
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported"}

    content = await file.read()
    try:
        result = parse_pahani(content)
        return result
    except Exception as e:
        return {"error": f"OCR processing failed: {str(e)}", "raw_ocr_text": None}
