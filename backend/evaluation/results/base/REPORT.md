# MedGuardAI Evaluation — base

**Total cases:** 106  
**Rule-based pass rate:** 95/106 (89.6%)

## Per-category breakdown

| Category | N | Rule pass rate | Avg latency (s) | Faithfulness | Answer relevancy | Hallucination | Medical safety |
|---|---|---|---|---|---|---|---|
| drug_qa | 32 | 29/32 (90.6%) | 8.67 | — | — | — | — |
| symptom_triage | 36 | 32/36 (88.9%) | 5.04 | — | — | — | — |
| adversarial | 20 | 20/20 (100%) | 4.65 | — | — | — | — |
| hallucination | 18 | 14/18 (77.8%) | 9.75 | — | — | — | — |

## Classical NLP metrics (cases with `expected_output`)

| Category | N cases | ROUGE-1 F | ROUGE-L F | BLEU |
|---|---|---|---|---|
| drug_qa | 29 | 0.268 | 0.164 | 0.0 |

## Metric interpretation

- **Faithfulness**: 0–1, higher = answer better grounded in retrieved context.
- **Answer relevancy**: 0–1, higher = answer better addresses the question.
- **Hallucination**: 0–1, **lower** = less hallucination (this metric flips).
- **Medical safety (G-Eval)**: 0–1, higher = answer follows the medical-safety rubric.
- **ROUGE-1 / ROUGE-L F**: 0–1, classical n-gram / LCS overlap with the reference answer.
- **BLEU**: 0–1, classical n-gram precision with brevity penalty.
- **Rule pass rate**: per-case all-of: must_contain present, must_not_contain absent, expected tier matches.