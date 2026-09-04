"""
Agri-Mitra AI Chatbot — Streamlit Interface
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import asyncio
from app.services.rag_service import rag_service
from app.services.elevenlabs_service import elevenlabs_service

st.set_page_config(
    page_title="Agri-Mitra AI Co-Pilot",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0b120e; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .source-badge {
        display: inline-block;
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-right: 4px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.title("🌱 Agri-Mitra Co-Pilot")
    st.caption("Multilingual RAG Agronomic Advisor")
    
    language = st.selectbox(
        "Language / भाषा",
        ["English", "Hindi", "Kannada", "Telugu", "Tamil", "Marathi", "Bengali", "Gujarati", "Malayalam", "Punjabi", "Odia"],
        index=0
    )
    
    st.divider()
    st.subheader("🌾 Telemetry Context")
    crop = st.text_input("Current Crop", value="Rice (Paddy)")
    ph = st.number_input("Soil pH", value=6.5, step=0.1, min_value=0.0, max_value=14.0)
    moisture = st.number_input("Soil Moisture (%)", value=28.0, step=1.0)
    nitrogen = st.number_input("Nitrogen (mg/kg)", value=45.0, step=1.0)
    temperature = st.number_input("Field Temp (°C)", value=31.0, step=0.5)

    field_context = {
        "crop": crop,
        "ph": ph,
        "moisture": moisture,
        "nitrogen_mg_kg": nitrogen,
        "temp_c": temperature
    }

    st.divider()
    tts_enabled = st.checkbox("Enable Regional Voice Audio", value=True)

# Session state chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! 🌱 I am **Agri-Mitra Co-Pilot**, your AI agronomist backed by ICAR and FAO knowledge bases. Ask me about fertilizer doses, soil pH, irrigation schedules, or yield planning!",
            "citations": []
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            st.caption("📄 **Sources:** " + ", ".join(c.get("source", "") for c in msg["citations"]))

# Chat Input
if prompt := st.chat_input("Ask about fertilizer, irrigation, pest management..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting ICAR & FAO knowledge engine..."):
            result = rag_service.query(
                question=prompt,
                language=language,
                field_context=field_context
            )
            answer = result.get("answer", "Data Not Available")
            citations = result.get("citations", [])

            st.markdown(answer)
            if citations:
                st.caption("📄 **Sources:** " + ", ".join(c.get("source", "") for c in citations))

            if tts_enabled:
                audio_bytes, _ = asyncio.run(elevenlabs_service.synthesize_speech(answer, language=language))
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mpeg")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })
