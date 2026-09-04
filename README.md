# 🌱 Agri-Mitra AI Chatbot & Agronomic Co-Pilot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-orange.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, multilingual AI Agronomic Chatbot and Co-Pilot tailored for Indian farmers and agricultural advisors. Built with **Retrieval-Augmented Generation (RAG)** over ICAR and FAO agronomic knowledge repositories, live field telemetry context enrichment, multilingual voice synthesis (ElevenLabs + Regional Neural Streams), and land record document OCR.

---

## 🚀 Key Features

- **🌾 Hybrid RAG Knowledge Engine**:
  - Leverages **Google Gemini + LangChain + ChromaDB** vector embeddings over ICAR/FAO agronomy documents.
  - Seamlessly falls back to an intelligent, verified offline agronomic rule engine covering soil pH, NPK fertilization schedules, irrigation timing, and crop recommendations.
- **🗣️ Multilingual & Regional Voice (11 Languages)**:
  - English, Hindi (हिंदी), Kannada (ಕನ್ನಡ), Telugu (తెలుగు), Tamil (தமிழ்), Marathi (मराठी), Bengali (বাংলা), Gujarati (ગુજરાતી), Malayalam (മലയാളം), Punjabi (ਪੰਜਾਬੀ), Odia (ଓଡ଼ିଆ).
  - Speech-to-Text input via Web Speech API.
  - High-fidelity Text-to-Speech (TTS) audio streaming via ElevenLabs Multilingual V2 with neural regional fallback.
- **📊 Real-time Field Telemetry Context**:
  - Contextually enriches prompts with live soil moisture, NPK sensor levels, field temperature, and crop stage.
- **📄 Land Record Document OCR**:
  - Automatically parses RTC/Pahani/7-12 land records to extract owner name, survey number, area in regional units (Guntha, Bigha, Cent, Acre to Hectare conversion), and soil taxonomy.
- **💻 Multiple Interfaces**:
  - **Standalone Web UI**: Dark-mode glassmorphism interface with voice input, audio playback, and telemetry simulator (`http://localhost:8000`).
  - **FastAPI REST API**: Fully documented Swagger/OpenAPI endpoints (`/docs`).
  - **Streamlit Interface**: Interactive Python frontend (`streamlit run streamlit_app.py`).

---

## 📁 Repository Structure

```
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py              # Application configuration & env management
│   ├── routers/
│   │   ├── __init__.py
│   │   └── copilot.py             # Chatbot RAG query, quick prompts, TTS, OCR endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── rag_service.py         # RAG agronomic reasoning & knowledge engine
│   │   ├── elevenlabs_service.py  # Regional multilingual voice synthesis
│   │   └── ocr_service.py         # Land document / Pahani OCR parser
│   ├── __init__.py
│   └── main.py                    # FastAPI server entry point
├── rag_docs/                      # ICAR & FAO knowledge PDFs
│   ├── AgriMitra_Entity_Normalization_Reference.pdf
│   ├── ICAR_Irrigation_Water_Management.pdf
│   ├── ICAR_Nitrogen_Management_Guidelines.pdf
│   ├── ICAR_Pest_Disease_Management_Advisory.pdf
│   ├── ICAR_Soil_Health_pH_Management.pdf
│   └── ICAR_Yield_Benchmarks_Crop_Statistics.pdf
├── static/
│   └── index.html                 # Interactive Web Chatbot UI
├── tests/
│   └── test_copilot_conversational.py # Test suite
├── .env.example                   # Environment template
├── .gitignore
├── requirements.txt               # Dependencies
├── streamlit_app.py               # Streamlit application
└── README.md
```

---

## 🛠️ Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/shravanth819/chatbot.git
cd chatbot

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` to provide your API keys (optional — the engine includes a full offline knowledge base fallback):
```env
GOOGLE_API_KEY=your_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

### 3. Run the Chatbot

#### Option A: FastAPI Web App & REST API (Recommended)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser for the Web Chat interface, or **`http://localhost:8000/docs`** for interactive API documentation.

#### Option B: Streamlit UI
```bash
streamlit run streamlit_app.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/copilot/query` | Submit question with language & telemetry context |
| `GET` | `/api/v1/copilot/quick-prompts` | Retrieve localized quick prompt chips |
| `POST` | `/api/v1/copilot/tts` | Synthesize regional audio stream (`audio/mpeg`) |
| `GET` | `/api/v1/copilot/tts/status` | Check voice engine status |
| `POST` | `/api/v1/ocr/pahani` | Extract structured data from Pahani/RTC PDF |
| `GET` | `/health` | Service health status |

---

## 🧪 Running Tests

```bash
pytest tests/
# or
python tests/test_copilot_conversational.py
```

---

## 📄 License
This project is licensed under the MIT License.