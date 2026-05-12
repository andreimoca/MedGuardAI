"""Tests for the human-feedback (RLHF data collection) endpoints."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fastapi.testclient import TestClient

import api.main as main_module
from api.main import app

client = TestClient(app)


def _redirect_feedback(tmp_path):
    """Point the feedback files at a temp dir so tests don't touch real data."""
    fb_dir = os.path.join(str(tmp_path), "feedback")
    main_module.FEEDBACK_DIR = fb_dir
    main_module.FEEDBACK_PATH = os.path.join(fb_dir, "feedback.jsonl")
    main_module.FEEDBACK_STUB_PATH = os.path.join(fb_dir, "feedback_dpo_stub.jsonl")
    return fb_dir


class TestFeedbackEndpoint:
    def test_thumbs_up_persisted(self, tmp_path):
        _redirect_feedback(tmp_path)
        resp = client.post("/api/v1/feedback", json={
            "query": "Is ibuprofen ok for a headache?",
            "answer": "Yes, follow the package dose.",
            "rating": "up",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["stored"] == 1
        with open(main_module.FEEDBACK_PATH, encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert row["rating"] == "up"
        assert row["query"].startswith("Is ibuprofen")
        assert row["correction"] is None

    def test_thumbs_down_with_correction(self, tmp_path):
        _redirect_feedback(tmp_path)
        client.post("/api/v1/feedback", json={
            "query": "my head hurts",
            "patient_context": {"age": 30, "weight": 70, "allergies": [], "conditions": []},
            "answer": "Take 800 mg ibuprofen every 4 hours.",
            "rating": "down",
            "correction": "Ask how long it's been going on first, then suggest OTC options at labeled doses.",
            "status": "success",
        })
        with open(main_module.FEEDBACK_PATH, encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert row["rating"] == "down"
        assert "Ask how long" in row["correction"]
        assert row["patient_context"]["age"] == 30

    def test_invalid_rating_rejected(self, tmp_path):
        _redirect_feedback(tmp_path)
        resp = client.post("/api/v1/feedback", json={
            "query": "x", "answer": "y", "rating": "maybe",
        })
        assert resp.status_code == 422

    def test_export_returns_dpo_stubs(self, tmp_path):
        _redirect_feedback(tmp_path)
        client.post("/api/v1/feedback", json={"query": "q1", "answer": "good answer", "rating": "up"})
        client.post("/api/v1/feedback", json={"query": "q2", "answer": "bad answer", "rating": "down"})
        client.post("/api/v1/feedback", json={
            "query": "q3", "answer": "bad answer 3", "rating": "down", "correction": "better answer 3",
        })
        resp = client.get("/api/v1/feedback/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2  # only the two thumbs-down
        sources = {it["meta"]["source"] for it in data["items"]}
        assert "feedback:downvote_only" in sources
        assert "feedback:user_correction" in sources
        for it in data["items"]:
            assert it["rejected"]
            if it["meta"]["source"] == "feedback:user_correction":
                assert it["chosen"] == "better answer 3"
            else:
                assert it["chosen"] is None
        assert os.path.exists(main_module.FEEDBACK_STUB_PATH)

    def test_export_empty_when_no_feedback(self, tmp_path):
        _redirect_feedback(tmp_path)
        resp = client.get("/api/v1/feedback/export")
        assert resp.status_code == 200
        assert resp.json() == {"count": 0, "items": []}
