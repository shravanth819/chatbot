"""
Agri-Mitra — ElevenLabs Conversational AI & Regional Voice TTS Service
Converts AI Co-Pilot agronomic answers into natural human voice in regional Indian languages.
Uses ElevenLabs Multilingual V2 (eleven_multilingual_v2) when an API key is provided,
and seamlessly falls back to high-fidelity regional neural audio streaming for Indian languages
(Kannada, Hindi, Telugu, Tamil, Marathi, Bengali, Gujarati, Malayalam, Punjabi, Odia, English).
"""
import re
import urllib.parse
import httpx
import logging
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

LANG_CODE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Telugu": "te",
    "Tamil": "ta",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Odia": "or",
}

# Clean markdown artifacts, citations, and formulas for fluent speech
def clean_text_for_voice(text: str) -> str:
    cleaned = re.sub(r"\[Source:[^\]]+\]", "", text)  # Remove citations
    cleaned = re.sub(r"https?://\S+", "", cleaned)      # Remove URLs
    cleaned = re.sub(r"[*#_`~>•]", "", cleaned)         # Remove markdown symbols
    cleaned = re.sub(r"\$[^$]+\$", "", cleaned)         # Remove LaTeX math
    cleaned = re.sub(r"[-]{2,}", "", cleaned)           # Remove separator dashes
    cleaned = re.sub(r"\s+", " ", cleaned).strip()      # Normalize spaces
    # Truncate text for single-utterance speech synthesis (up to ~350 chars for crisp speech)
    if len(cleaned) > 350:
        first_sentence = cleaned[:350].rsplit(".", 1)[0]
        cleaned = first_sentence + "." if first_sentence else cleaned[:350]
    return cleaned


class ElevenLabsService:
    @staticmethod
    def is_configured() -> bool:
        return bool(settings.ELEVENLABS_API_KEY and len(settings.ELEVENLABS_API_KEY.strip()) > 5)

    @staticmethod
    async def synthesize_speech(
        text: str,
        voice_id: str | None = None,
        model_id: str | None = None,
        language: str = "English",
        custom_api_key: str | None = None
    ) -> tuple[bytes | None, str | None]:
        """
        Synthesizes speech in the requested regional language.
        1. Tries ElevenLabs Multilingual V2 if API key is present.
        2. Seamlessly falls back to regional TTS audio streaming in Kannada, Telugu, Tamil, Marathi, Hindi, etc.
        """
        cleaned_text = clean_text_for_voice(text)
        if not cleaned_text:
            return None, "No text to synthesize"

        api_key = custom_api_key or settings.ELEVENLABS_API_KEY

        # 1. Primary: ElevenLabs Multilingual V2
        if api_key and len(api_key.strip()) > 5:
            selected_voice = voice_id or settings.ELEVENLABS_VOICE_ID or "21m00Tcm4TlvDq8ikWAM"
            selected_model = model_id or settings.ELEVENLABS_MODEL_ID or "eleven_multilingual_v2"

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice}"
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"
            }
            payload = {
                "text": cleaned_text,
                "model_id": selected_model,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "style": 0.1,
                    "use_speaker_boost": True
                }
            }

            try:
                async with httpx.AsyncClient(timeout=25.0) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    if response.status_code == 200 and len(response.content) > 500:
                        return response.content, None
                    else:
                        logger.warning(f"ElevenLabs returned {response.status_code}, falling back to regional TTS...")
            except Exception as e:
                logger.error(f"ElevenLabs TTS failed ({e}), falling back to regional TTS...")

        # 2. Resilient Fallback: High-quality Regional Neural Audio Stream
        lang_code = LANG_CODE_MAP.get(language, "en")
        try:
            encoded_query = urllib.parse.quote(cleaned_text)
            tts_url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded_query}&tl={lang_code}&client=tw-ob"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(tts_url, headers=headers)
                if res.status_code == 200 and len(res.content) > 500:
                    return res.content, None
        except Exception as e:
            logger.error(f"Regional fallback TTS stream failed: {e}")

        return None, "Failed to generate regional voice audio"


elevenlabs_service = ElevenLabsService()
