"""Tests for the SFT dataset builder (MedQuAD rewrite + curated safety-triage themes)."""
import asyncio
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

import build_qa_dataset as bqd


SFT_ROW_KEYS = {"task_type", "drug", "section", "question", "answer", "source", "source_id", "meta"}


class TestSymptomRows:
    def setup_method(self):
        self.rows = bqd.build_symptom_rows(questions_per_theme=2)

    def test_nonempty_and_schema(self):
        assert len(self.rows) > 100
        for r in self.rows:
            assert SFT_ROW_KEYS.issubset(r.keys())
            assert r["task_type"] == "symptom"
            assert r["question"] and r["answer"]
            assert r["section"] in {"low", "medium", "high_emergency"}
            assert r["meta"]["source_dataset"].startswith("MedGuardAI curated")

    def test_deterministic(self):
        again = bqd.build_symptom_rows(questions_per_theme=2)
        assert [bqd.row_key(r) for r in self.rows] == [bqd.row_key(r) for r in again]
        assert [r["answer"] for r in self.rows] == [r["answer"] for r in again]

    def test_emergency_answer_has_tag_low_does_not(self):
        em = [r for r in self.rows if r["section"] == "high_emergency"]
        low = [r for r in self.rows if r["section"] == "low"]
        assert em and low
        assert all("[EMERGENCY]" in r["answer"] for r in em)
        assert all("[EMERGENCY]" not in r["answer"] for r in low)

    def test_emergency_answers_recommend_no_medication(self):
        for r in self.rows:
            if r["section"] == "high_emergency":
                lowered = r["answer"].lower()
                assert "ibuprofen" not in lowered
                assert "acetaminophen" not in lowered

    def test_medium_answers_have_clinician_referral_no_emergency_tag(self):
        for r in self.rows:
            if r["section"] == "medium":
                assert "[EMERGENCY]" not in r["answer"]
                assert "see a doctor" in r["answer"]


class TestMedQuADTasks:
    def _write_csv(self, tmp_path):
        p = tmp_path / "medquad.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "answer", "source", "focus_area"])
            w.writerow(["What is aspirin?", "Aspirin is a pain reliever.", "MedlinePlus", "Aspirin"])
            w.writerow(["What is aspirin?", "Aspirin is a pain reliever.", "MedlinePlus", "Aspirin"])  # dup
            w.writerow(["What are signs of a stroke?", "Sudden weakness, trouble speaking.", "NINDS", "Stroke"])
            w.writerow(["", "no question here", "x", "y"])  # dropped (empty question)
        return str(p)

    def test_parsing_dedup_and_schema(self, tmp_path):
        tasks = bqd.build_medquad_tasks(max_rows=10, seed=1, csv_path=self._write_csv(tmp_path))
        assert len(tasks) == 2  # duplicate collapsed, empty-question row dropped
        for t in tasks:
            assert t.task_type == "medquad_qa"
            assert t.source_id.startswith("medquad::")
            assert t.extras["source_dataset"].startswith("MedQuAD (Kaggle:")
        aspirin = next(t for t in tasks if "aspirin" in t.question.lower())
        # the gen prompt is built ENTIRELY from the source answer — no invented facts
        assert "Aspirin is a pain reliever." in aspirin.gen_prompt()

    def test_max_rows_cap(self, tmp_path):
        tasks = bqd.build_medquad_tasks(max_rows=1, seed=1, csv_path=self._write_csv(tmp_path))
        assert len(tasks) == 1

    def test_run_task_rewrite_then_judge_yes(self, tmp_path, monkeypatch):
        task = bqd.build_medquad_tasks(max_rows=10, seed=1, csv_path=self._write_csv(tmp_path))[0]

        calls = {"n": 0}

        async def fake_call_chat(client, model, prompt, temperature):
            calls["n"] += 1
            if "Reply with EXACTLY one word" in prompt:  # the faithfulness judge
                return "YES"
            return "Here is a safe rewrite. Please consult a healthcare professional."

        monkeypatch.setattr(bqd, "call_chat", fake_call_chat)
        row = asyncio.run(bqd.run_task(task, client=None, model="x", judge=True))
        assert row is not None
        assert SFT_ROW_KEYS.issubset(row.keys())
        assert row["task_type"] == "medquad_qa"
        assert row["answer"].startswith("Here is a safe rewrite")
        assert row["source"] == task.source_answer
        assert calls["n"] == 2

    def test_run_task_judge_no_drops_the_row(self, tmp_path, monkeypatch):
        task = bqd.build_medquad_tasks(max_rows=10, seed=1, csv_path=self._write_csv(tmp_path))[0]

        async def fake_call_chat(client, model, prompt, temperature):
            if "Reply with EXACTLY one word" in prompt:
                return "NO"
            return "A rewrite that is long enough to pass the length check."

        monkeypatch.setattr(bqd, "call_chat", fake_call_chat)
        assert asyncio.run(bqd.run_task(task, client=None, model="x", judge=True)) is None
