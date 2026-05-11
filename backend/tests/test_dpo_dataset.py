"""Tests for the DPO preference-dataset builder (Phase 3 / RLHF)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation")))

import build_dpo_dataset as bdd


REQUIRED_ROW_KEYS = {"system", "prompt", "chosen", "rejected", "weight", "meta"}


class TestSeededNegatives:
    def setup_method(self):
        self.items = bdd.build_seeded_negatives(seed=123)

    def test_nonempty_and_deterministic(self):
        again = bdd.build_seeded_negatives(seed=123)
        assert len(self.items) > 50
        assert [i.key() for i in self.items] == [i.key() for i in again]

    def test_rows_have_schema(self):
        for it in self.items:
            row = it.to_row()
            assert REQUIRED_ROW_KEYS.issubset(row.keys())
            assert row["system"] == bdd.SYSTEM_PROMPT
            assert isinstance(row["prompt"], str) and row["prompt"].startswith("Drug:")
            assert row["chosen"] and row["rejected"]
            assert row["chosen"] != row["rejected"]
            assert "key" in row["meta"] and "source" in row["meta"]

    def test_emergency_pairs_have_tag_on_chosen_only(self):
        em = [i for i in self.items if i.meta.get("triage") == "high_emergency"]
        assert em, "expected some high_emergency seeded pairs"
        for it in em:
            assert "[EMERGENCY]" in it.chosen
            assert "[EMERGENCY]" not in it.rejected

    def test_low_tier_over_triage_negatives_present(self):
        # Some low-tier pairs reject an unwarranted [EMERGENCY] escalation.
        low_over = [i for i in self.items
                    if i.meta.get("triage") == "low" and "[EMERGENCY]" in i.rejected]
        assert low_over
        for it in low_over:
            assert "[EMERGENCY]" not in it.chosen

    def test_prescription_bait_pairs_present(self):
        rx = [i for i in self.items if i.meta.get("source") == "seeded:prescription_by_name"]
        assert len(rx) >= 3
        for it in rx:
            # rejected names a specific Rx drug; chosen defers to a clinician.
            assert "prescription-only" in it.chosen or "clinician" in it.chosen


class TestUserMsg:
    def test_format(self):
        assert bdd.user_msg("naproxen", "What dose?") == "Drug: naproxen\nQuestion: What dose?"
        assert bdd.user_msg("", "x") == "Drug: n/a\nQuestion: x"


class TestRuleCheckMustContainAny:
    """Covers the new must_contain_any semantics added to run_eval.check_rule_based."""
    def test_any_satisfied(self):
        import run_eval
        case = {"must_contain_any": ["no information", "not found", "do not have"]}
        assert run_eval.check_rule_based(case, "Sorry, I do not have that information.")["rule_passed"] is True

    def test_any_not_satisfied(self):
        import run_eval
        case = {"must_contain_any": ["no information", "not found"]}
        r = run_eval.check_rule_based(case, "The dose is 500 mg twice daily.")
        assert r["rule_passed"] is False
        assert r["must_contain_any_satisfied"] is False

    def test_empty_any_is_passthrough(self):
        import run_eval
        case = {"must_contain": ["ibuprofen"]}
        assert run_eval.check_rule_based(case, "Take ibuprofen as directed.")["rule_passed"] is True
