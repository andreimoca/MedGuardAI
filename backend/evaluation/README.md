# MedGuardAI — Phase 4: Evaluation

End-to-end automated evaluation of the agent. Uses [DeepEval](https://github.com/confident-ai/deepeval)
for LLM-as-judge metrics plus rule-based checks for specific safety patterns.

## What gets measured

Per test case, we collect:

| Signal | Type | Source |
|---|---|---|
| `must_contain` / `must_not_contain` | rule-based | hand-curated per case |
| Expected tier (low / medium / high_emergency) | rule-based | hand-curated per case |
| Latency | runtime | measured |
| Faithfulness (0-1) | LLM judge | DeepEval — is the answer grounded in retrieved context? |
| Answer relevancy (0-1) | LLM judge | DeepEval — does the answer address the question? |
| Hallucination (0-1) | LLM judge | DeepEval — does the answer make up facts? |
| Medical safety (0-1) | LLM judge (G-Eval) | custom rubric — covers tier accuracy, no Rx-by-name, no fabricated dosing |

## Held-out eval set

`data/eval_set.jsonl` — 100 hand-curated cases (not LLM-generated) across:
- **drug_qa** (20 cases): dosage, contraindications, interactions, allergy cross-reactivity, pediatric, geriatric, pregnancy
- **symptom_triage** (24 cases): vague & specific queries spanning low / medium / high_emergency tiers
- **adversarial** (10 cases): prompt-injection, off-topic, dangerous requests, contradictions
- **hallucination** (10 cases): non-existent drugs, fabricated conditions, off-label invention probes

To regenerate or extend the set, edit `build_eval_set.py` and run:
```bash
python evaluation/build_eval_set.py
```

## LLM-as-judge backend

The judge LLM is picked by env var, in priority order:
1. **DeepSeek** (`DATASET_LLM_URL` + `DATASET_LLM_KEY`) — strongly preferred for stronger, methodologically-independent judging.
2. **Local LM Studio** (`LOCAL_LLM_URL`) — fallback if no DeepSeek key.

Using the same fine-tuned Gemma as judge of itself is circular and gives unreliable scores — DeepSeek-V3 is a much better evaluator.

## Running

```bash
cd backend

# 1. Generate the eval set (one-time)
python evaluation/build_eval_set.py

# 2. Run the suite (label your run for later comparison)
python evaluation/run_eval.py --label base

# 3. After Phase 2 (fine-tune), run again
python evaluation/run_eval.py --label sft

# 4. After Phase 3 (DPO), run again
python evaluation/run_eval.py --label sft-dpo

# Quick smoke run (10 cases, no LLM judge cost)
python evaluation/run_eval.py --limit 10 --skip-deepeval --label smoke
```

## Output layout

```
evaluation/results/<label>/
  cases.jsonl    # per-case results (query, response, scores, latency)
  summary.json   # aggregate scores per category
  REPORT.md      # human-readable summary table
```

`REPORT.md` is the file you commit and show to the grader.

## Cost

- Rule-based-only run: $0 (no LLM judge calls).
- Full DeepEval run with DeepSeek as judge: ~$0.30–$0.60 per 100-case run (each test case → ~4 judge calls × ~$0.001).

If your DeepSeek balance is tight, use `--skip-deepeval` to get the rule-based table for free, and reserve the paid run for the final comparison numbers in your report.

## Comparison across runs

After running with multiple labels (e.g. `base`, `sft`, `sft-dpo`), eyeball
`results/*/summary.json` side-by-side. The key columns to show in the final
project report:

| | base | SFT | SFT+DPO |
|---|---|---|---|
| Symptom tier accuracy | x% | y% | z% |
| Faithfulness | x | y | z |
| Hallucination (lower = better) | x | y | z |
| Medical safety (G-Eval) | x | y | z |
| Rx-only mentions (rule violations) | x | y | z |

Improvement across the three columns is what you defend in the milestone deliverable.
