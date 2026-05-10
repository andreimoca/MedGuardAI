# MedGuardAI Evaluation — smoke

**Total cases:** 10  
**Rule-based pass rate:** 5/10 (50.0%)

## Per-category breakdown

| Category | N | Rule pass rate | Avg latency (s) | Faithfulness | Answer relevancy | Hallucination | Medical safety |
|---|---|---|---|---|---|---|---|
| drug_qa | 10 | 5/10 (50%) | 12.5 | — | — | — | — |

## Classical NLP metrics (cases with `expected_output`)

| Category | N cases | ROUGE-1 F | ROUGE-L F | BLEU |
|---|---|---|---|---|
| drug_qa | 1 | 0.082 | 0.082 | 0.0 |

## Metric interpretation

- **Faithfulness**: 0–1, higher = answer better grounded in retrieved context.
- **Answer relevancy**: 0–1, higher = answer better addresses the question.
- **Hallucination**: 0–1, **lower** = less hallucination (this metric flips).
- **Medical safety (G-Eval)**: 0–1, higher = answer follows the medical-safety rubric.
- **ROUGE-1 / ROUGE-L F**: 0–1, classical n-gram / LCS overlap with the reference answer.
- **BLEU**: 0–1, classical n-gram precision with brevity penalty.
- **Rule pass rate**: per-case all-of: must_contain present, must_not_contain absent, expected tier matches.