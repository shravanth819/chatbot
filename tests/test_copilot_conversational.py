"""
Agri-Mitra AI Chatbot — Test Suite
Tests conversational greetings, agronomic queries, RAG context enrichment, and language localization.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag_service import rag_service


def test_greeting_english():
    res = rag_service.query("Hello", language="English")
    assert "Agri-Mitra" in res["answer"]
    assert res["confidence"] == "High"


def test_greeting_hindi():
    res = rag_service.query("नमस्ते", language="Hindi")
    assert "कृषि-मित्र" in res["answer"]


def test_agronomic_query_with_telemetry():
    field_context = {
        "crop_type": "Rice",
        "ph": 5.2,
        "moisture": 25.0,
        "nitrogen": 35.0
    }
    res = rag_service.query("How much nitrogen should I apply?", language="English", field_context=field_context)
    assert res["answer"] is not None
    assert len(res["answer"]) > 10


def test_offline_knowledge_soil_ph():
    res = rag_service.query("Is pH 6.5 good for rice?", language="English")
    assert res["answer"] is not None
    assert len(res["answer"]) > 10


if __name__ == "__main__":
    test_greeting_english()
    test_greeting_hindi()
    test_agronomic_query_with_telemetry()
    test_offline_knowledge_soil_ph()
    print("[SUCCESS] All chatbot tests passed successfully!")
