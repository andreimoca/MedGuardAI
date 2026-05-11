# MedGuardAI — Phase 2: Fine-tuning Gemma-3-4b

Two-stage workflow. Stage 1 happens locally (uses LM Studio for inference, no
heavy training). Stage 2 happens on a free Colab/Kaggle T4 (16 GB VRAM is
plenty; your 6 GB laptop GPU isn't).

## Stage 1 — Generate the synthetic SFT dataset

Two task types feed the same JSONL output:

1. **drug-labels** — for each FDA label section, generate Q&A pairs grounded in
   the section text, then validate them with a judge prompt.
2. **symptoms** — for each entry in a curated symptom triage list (headache,
   back pain, dizziness, chest pain, anaphylaxis, overdose, etc.), generate
   user-style queries with safety-aware responses. Emergency-tier symptoms
   produce strict `[EMERGENCY] ...` responses; lower-tier may recommend OTC
   drugs from our FDA corpus; medium-tier defers to a clinician.

### Backend selection

The dataset builder picks its LLM separately from the runtime agent. Set these
in `.env`:

```
DATASET_LLM_URL=https://api.deepseek.com/v1
DATASET_LLM_KEY=sk-...
DATASET_LLM_MODEL=deepseek-chat
```

If `DATASET_LLM_KEY` is empty, the script falls back to your local LM Studio
(`LOCAL_LLM_URL`). DeepSeek is **~10–30× faster** thanks to async concurrency
(local LM Studio serializes requests) and costs ~$5–10 USD for the full
~12k-call run. The fine-tuned model still runs locally — only the *training
data generation* uses the remote API, which is the standard pattern for
SLM fine-tuning (Alpaca, Vicuna, Orca, Phi, etc.).

### Running

```bash
cd backend
# Quick smoke run (~5 drugs + 30 symptom themes, ~3-5 min on DeepSeek)
python training/build_qa_dataset.py --max-drugs 5 --concurrency 10

# Real run for SFT training (~3-5k pairs, ~10-20 min on DeepSeek)
python training/build_qa_dataset.py --max-drugs 1000 --concurrency 20

# Drug labels only (skip symptoms)
python training/build_qa_dataset.py --task drug-labels --max-drugs 1000

# Symptoms only
python training/build_qa_dataset.py --task symptoms

# Skip the validator (~2x faster, lower quality)
python training/build_qa_dataset.py --max-drugs 1000 --no-judge
```

Output: `training/data/sft_pairs.jsonl`. The script is **resumable** — re-run
it after a crash and it picks up where it left off (skips any
`(task_type, drug, section, source_id)` tuples already in the file).

Each row:
```json
{
  "task_type": "drug_label",
  "drug": "Naproxen",
  "section": "drug_interactions",
  "question": "Can I take naproxen with warfarin?",
  "answer": "Naproxen and warfarin have a synergistic effect on bleeding...",
  "source": "<the FDA section text>",
  "source_id": "naproxen_8c45ef1f-....json"
}
```

For `task_type: "symptom"` rows, `section` is the triage tier
(`low` / `medium` / `high_emergency`) and an extra `meta` field carries the
theme label and notes.

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
# 1. build the preference dataset
#    sources: seeded safety hard-negatives (deterministic) + degraded SFT answers (DeepSeek + position-swap-consistent judge) + collected human feedback
python training/build_dpo_dataset.py --candidate-mode none                       # seeded negatives only (zero API cost)
python training/build_dpo_dataset.py --prompts-from sft --n-prompts 300 --concurrency 15
python training/build_dpo_dataset.py --prompts-from feedback --include-seeded     # fold in UI thumbs-down + corrections
# -> training/data/dpo_pairs.jsonl   (resumable; human-feedback pairs are weighted up)
```

Then open [`finetune_gemma_dpo.ipynb`](finetune_gemma_dpo.ipynb) in Colab (T4), upload `dpo_pairs.jsonl` and the `medguard-sft/` adapter folder, run top-to-bottom (~20–40 min):

| Artifact | Purpose |
|---|---|
| `medguard-dpo/` | DPO LoRA adapter |
| `medguard-dpo-gguf/medguard-gemma-3-4b-dpo-q4_k_m.gguf` | drop-in for LM Studio |

Deploy as in Stage 3 (`export_to_lmstudio.py copy --gguf …`), set `LOCAL_LLM_MODEL=medguard/medguard-gemma-3-4b-dpo`, restart the backend, then run the eval comparison: `backend/evaluation/run_eval.py --label {base,sft,dpo} --model-override …` × 3, then `backend/evaluation/compare_runs.py --runs base sft dpo --per-case`.

Human feedback feeds back in: the running app appends thumbs-up/down (and optional corrections) to `backend/data/feedback/feedback.jsonl` via `POST /api/v1/feedback`; `build_dpo_dataset.py --prompts-from feedback` turns the thumbs-down items into preference pairs.

## Files

- `build_qa_dataset.py` — synthetic Q&A generator with judge-based validation (Phase 2 SFT data)
- `finetune_gemma_lora.ipynb` — Colab/Kaggle QLoRA SFT notebook
- `build_dpo_dataset.py` — preference-dataset builder (seeded negatives + degraded SFT answers + human feedback)
- `finetune_gemma_dpo.ipynb` — Colab/Kaggle DPO notebook (continues from the SFT adapter)
- `export_to_lmstudio.py` — copy GGUF locally / push to HF Hub
- `rebalance_dataset.py` — rebalance `sft_pairs.jsonl` across task types / triage tiers
- `data/sft_pairs.jsonl`, `data/dpo_pairs.jsonl` — generated datasets (gitignored)
- `adapters/` — local copies of trained adapters (gitignored)
