# MedGuardAI — Phase 2: Fine-tuning Gemma-3-4b

Two-stage workflow. Stage 1 happens locally (uses LM Studio for inference, no
heavy training). Stage 2 happens on a free Colab/Kaggle T4 (16 GB VRAM is
plenty; your 6 GB laptop GPU isn't).

## Stage 1 — Build the SFT dataset (MedQuAD + curated safety-triage themes)

`build_qa_dataset.py` writes one JSONL with two kinds of rows:

1. **medquad** — [MedQuAD](https://github.com/abachaa/MedQuAD), the Medical
   Question Answering Dataset (consumer-health Q&A curated from U.S. National
   Institutes of Health websites), pulled from Kaggle with `kagglehub`
   (`pythonafroz/medquad-medical-question-answer-for-ai-research`). For each
   Q&A pair the builder LLM **rewrites the existing answer** into MedGuardAI's
   safety-first voice (concise, "consult a healthcare provider", `[EMERGENCY]`
   only for true emergencies, never invents a dose or a drug name). A second
   call judges the rewrite for **faithfulness to the original MedQuAD answer** —
   the LLM reformats, it does not introduce new clinical facts. Rows that fail
   the judge are dropped. (`task_type: "medquad_qa"`.)
2. **symptoms** — for each entry in `SYMPTOM_THEMES` (a hand-curated list of
   ~140 symptom-triage scenarios: headache, back pain, dizziness, chest pain,
   anaphylaxis, overdose, …) the safety-aware answer is rendered from a
   **deterministic template** keyed on the triage tier — emergency tiers get a
   strict `[EMERGENCY] …` response and recommend no medication; medium tier
   gives brief OTC guidance and defers to a clinician within a timeframe; low
   tier asks a clarifying question first. **No LLM is used for this slice.**
   (`task_type: "symptom"`, `section` = `low` / `medium` / `high_emergency`.)

### Backend selection (only the MedQuAD rows touch the network)

`--task symptoms` runs fully offline. For the MedQuAD rewrite/judge, the
builder picks its LLM separately from the runtime agent — set in `.env`:

```
DATASET_LLM_URL=https://api.deepseek.com/v1
DATASET_LLM_KEY=sk-...
DATASET_LLM_MODEL=deepseek-chat
```

If `DATASET_LLM_KEY` is empty it falls back to your local LM Studio
(`LOCAL_LLM_URL`). DeepSeek is far faster here thanks to async concurrency
(local LM Studio serializes requests); reformatting ~4k MedQuAD answers with
the faithfulness judge costs roughly a couple of USD. `kagglehub` downloads the
public dataset; if Kaggle asks for credentials set `KAGGLE_USERNAME` /
`KAGGLE_KEY` (or drop `~/.kaggle/kaggle.json` in place).

### Running

```bash
cd backend
pip install kagglehub

# Smoke run: ~30 MedQuAD rows reformatted + all the deterministic symptom themes
python training/build_qa_dataset.py --task all --max-rows 30 --concurrency 10

# Real run for SFT training (resumable; ~a couple $ on DeepSeek)
python training/build_qa_dataset.py --task all --max-rows 4000 --concurrency 20

# Deterministic symptom-triage rows only (offline, no API key needed)
python training/build_qa_dataset.py --task symptoms

# MedQuAD rows only, skipping the faithfulness judge (~2x faster, lower quality)
python training/build_qa_dataset.py --task medquad --max-rows 4000 --no-judge
```

Output: `training/data/sft_pairs.jsonl`. The script is **resumable** — re-run
it after a crash and it picks up where it left off (skips any
`(task_type, drug, section, source_id)` tuples already in the file).

A MedQuAD row:
```json
{
  "task_type": "medquad_qa",
  "drug": "Glaucoma",
  "section": "informational",
  "question": "What are the treatments for Glaucoma?",
  "answer": "<the MedQuAD answer, rewritten into MedGuardAI's voice>",
  "source": "<the original MedQuAD answer text — ground truth for the judge>",
  "source_id": "medquad::1a2b3c4d5e6f7a8b",
  "meta": {
    "source_dataset": "MedQuAD (Kaggle: pythonafroz/medquad-medical-question-answer-for-ai-research)",
    "origin_source": "NIHSeniorHealth",
    "focus_area": "Glaucoma"
  }
}
```

A symptom row carries `section` = the triage tier and `meta` with the theme
label, triage tier, and clinical notes.

## Stage 2 — Train QLoRA on Colab/Kaggle (cloud)

1. Open [`finetune_gemma_lora.ipynb`](finetune_gemma_lora.ipynb) in
   [Google Colab](https://colab.research.google.com) (`File → Upload notebook`).
   In Colab make sure the runtime is **GPU → T4** (Runtime menu → Change runtime type).
2. In Colab, click the **Files** panel (left sidebar) and upload your
   `training/data/sft_pairs.jsonl`.
3. Run cells top-to-bottom. ~30–60 min for 1 epoch on ~3-5k pairs.
4. The last cells either download the artifacts as zip files or push them to a
   private Hugging Face Hub repo.

You'll get two artifacts:

| Artifact | Size | Purpose |
|---|---|---|
| `medguard-sft/` | ~80 MB | LoRA adapter — keep this for Phase 3 (DPO) |
| `medguard-gguf/medguard-gemma-3-4b-q4_k_m.gguf` | ~2.5 GB | Drop-in for LM Studio |

## Stage 3 — Use the fine-tuned model in MedGuardAI (local)

Once the GGUF is on this machine:

```bash
# Auto-copy the .gguf into LM Studio's models folder:
python training/export_to_lmstudio.py copy --gguf path/to/medguard-gemma-3-4b-q4_k_m.gguf

# Push everything to a private HF Hub for your team:
python training/export_to_lmstudio.py push \
    --repo your-username/medguardai-gemma-3-4b \
    --adapter-dir path/to/medguard-sft \
    --gguf-dir path/to/medguard-gguf \
    --token hf_xxx
```

Then in LM Studio, **load** the new model, and edit [.env](../.env):

```
LOCAL_LLM_MODEL=medguard/medguard-gemma-3-4b
```

Restart the backend (`python src/api/main.py` from `backend/`). The agent now
uses your fine-tuned weights — no application code change needed.

## Sharing with teammates

The Colab notebook's last cell pushes both the LoRA adapter and the GGUF to a
**private Hugging Face Hub repo**. Each teammate can then:

```bash
pip install huggingface_hub
huggingface-cli login   # paste a read token
huggingface-cli download your-username/medguardai-gemma-3-4b --include "gguf/*" --local-dir ./med-model
python training/export_to_lmstudio.py copy --gguf ./med-model/gguf/medguard-gemma-3-4b-q4_k_m.gguf
```

License note: Gemma-3 is distributed under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Redistribution is
permitted; pass the terms along to your teammates and don't use the model for
the prohibited harmful purposes.

## Stage 4 — Phase 3: DPO (preference / RLHF)

DPO continues from the **SFT LoRA adapter** (`medguard-sft/`, kept from Stage 2 — on Colab/Drive/HF; if it's truly gone, re-run Stage 2 or DPO from base Gemma).

```bash
cd backend
# 1. build the preference dataset (no LLM involved — fully deterministic + real human feedback)
#    sources: hand-authored safety hard-negatives + thumbs-down/correction pairs from the app
python training/build_dpo_dataset.py                # seeded negatives + human-feedback pairs
python training/build_dpo_dataset.py --no-feedback  # seeded negatives only
python training/build_dpo_dataset.py --no-seeded    # only the human-feedback pairs
# -> training/data/dpo_pairs.jsonl   (resumable; human-feedback pairs are weighted up)
```

Then open [`finetune_gemma_dpo.ipynb`](finetune_gemma_dpo.ipynb) in Colab (T4), upload `dpo_pairs.jsonl` and the `medguard-sft/` adapter folder, run top-to-bottom (~20–40 min):

| Artifact | Purpose |
|---|---|
| `medguard-dpo/` | DPO LoRA adapter |
| `medguard-dpo-gguf/medguard-gemma-3-4b-dpo-q4_k_m.gguf` | drop-in for LM Studio |

Deploy as in Stage 3 (`export_to_lmstudio.py copy --gguf …`), set `LOCAL_LLM_MODEL=medguard/medguard-gemma-3-4b-dpo`, restart the backend, then run the eval comparison: `backend/evaluation/run_eval.py --label {base,sft,dpo} --model-override …` × 3, then `backend/evaluation/compare_runs.py --runs base sft dpo --per-case`.

Human feedback feeds back in: the running app appends thumbs-up/down (and optional corrections) to `backend/data/feedback/feedback.jsonl` via `POST /api/v1/feedback`; `build_dpo_dataset.py` turns each thumbs-down item *that came with a user correction* into a preference pair (correction = `chosen`, downvoted answer = `rejected`).

## Files

- `build_qa_dataset.py` — SFT data builder: rewrites MedQuAD Q&A into our voice (with a faithfulness judge) + renders the curated safety-triage themes deterministically
- `finetune_gemma_lora.ipynb` — Colab/Kaggle QLoRA SFT notebook
- `build_dpo_dataset.py` — preference-dataset builder (hand-authored safety hard-negatives + human-correction pairs; no LLM)
- `finetune_gemma_dpo.ipynb` — Colab/Kaggle DPO notebook (continues from the SFT adapter)
- `export_to_lmstudio.py` — copy GGUF locally / push to HF Hub
- `rebalance_dataset.py` — trim over-represented symptom triage tiers in `sft_pairs.jsonl`
- `data/sft_pairs.jsonl`, `data/dpo_pairs.jsonl` — generated datasets (gitignored)
- `adapters/` — local copies of trained adapters (gitignored)
