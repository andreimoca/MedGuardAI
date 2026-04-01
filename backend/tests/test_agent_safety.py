import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add src to Python path for importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from api.main import app
from rag.agent import ClinicalAgent

client = TestClient(app)

class TestClinicalAgent:
    def setup_method(self):
        # We test the primitive check without needing the full LLM
        self.agent = ClinicalAgent()

    def test_emergency_guardrail_tripped(self):
        """Ensure severe symptoms trigger the immediate fallback mechanism."""
        assert self.agent.check_for_emergency("I cannot breathe and my chest hurts") == True
        assert self.agent.check_for_emergency("suspected overdose of paracetamol") == True
        assert self.agent.check_for_emergency("anaphylaxis reaction after taking amoxicillin") == True

    def test_emergency_guardrail_safe(self):
        """Ensure normal queries bypass the primitive guardrail."""
        assert self.agent.check_for_emergency("Is it safe to take ibuprofen for a headache?") == False
        assert self.agent.check_for_emergency("What is the standard dosage for vitamin C?") == False

    def test_patient_context_formatting(self):
        """Ensure that the context is formatted securely for prompt injection."""
        context = {
            "age": 30,
            "weight": 70,
            "allergies": ["Aspirin"],
            "conditions": ["Asthma"]
        }
        formatted = self.agent.format_patient_context(context)
        assert "Age: 30" in formatted
        assert "Weight: 70" in formatted
        assert "Aspirin" in formatted
        assert "Asthma" in formatted


class TestAPIEndpoints:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_emergency_endpoint(self):
        """Test the integration between the API and the agent guardrails."""
        payload = {
            "query": "I am having a heart attack",
            "patient_context": {
                "age": 60,
                "weight": 85,
                "allergies": [],
                "conditions": []
            }
        }
        response = client.post("/api/v1/ask", json=payload)
        
        # We expect a success 200 HTTP code, but an 'emergency' status in the JSON
        assert response.status_code == 200
        assert response.json()["status"] == "emergency"
        assert "[EMERGENCY GUARDRAIL TRIPPED]" in response.json()["answer"]
