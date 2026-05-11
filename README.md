# MedGuardAI

**Safety-First Medical Assistant powered by Agentic RAG**

## Team Information

**Team Name:** MEDAI
**Team Members:** Andrei Moca, Bogdan Borodi, Calin Pauliuc, Alexandra Petrea, Ana Vaicum

**GitHub Repository:** [https://github.com/andreimoca/MedGuardAI](https://github.com/andreimoca/MedGuardAI)

## Description of the Architecture

MedGuardAI is an AI-driven medical assistant powered by Agentic RAG. It utilizes a FastAPI Python backend to handle retrieving context and communicating with a **fine-tuned** Small Language Model (a QLoRA + DPO fine-tune of Google Gemma-3-4b, served locally via LM Studio) while maintaining strict guardrails and evaluating patient context (age, weight, conditions, allergies). The frontend is built as a Single Page Application (SPA) using React, Vite, and Framer Motion for a fluid Chat UX. The architecture ensures an absolute focus on safety and clinical accuracy by validating semantic queries against an internal Vector Database built from official drug leaflets, and it closes a **human-feedback flywheel**: thumbs-up/down + corrections collected in the UI feed the DPO preference dataset.

### The full pipeline (one diagram)

```
                    FDA DailyMed labels (OpenFDA)
                              │
              ┌───────────────┴───────────────┐
        chunk + embed                   synthetic Q&A gen (DeepSeek)
        (all-MiniLM-L6-v2)              build_qa_dataset.py
              │                                │
        ChromaDB vector store            sft_pairs.jsonl
              │                                │
              │                         QLoRA SFT  (finetune_gemma_lora.ipynb, T4)
              │                                │
              │                         DPO / RLHF (finetune_gemma_dpo.ipynb, T4)
              │                          ▲      │   ▲
              │                          │      ▼   │
   ┌──────────┴───────────┐    dpo_pairs.jsonl   medguard-…-dpo.gguf  (LM Studio)
   │   RAG retrieval (MMR) │    (build_dpo_dataset.py)        │
   └──────────┬───────────┘     ▲                            │
              │                 │ thumbs-down + corrections  │
              ▼                 │                            ▼
   input guard → LangGraph agent (6 tools) → output guard / groundedness → answer
                                                              │
                                                       React UI + 👍/👎 ──► feedback.jsonl
```

### Core Features

- **Safety-First Design:** Multi-layer guardrails prevent hallucinated medical advice
- **Grounded Responses:** All answers derived exclusively from FDA-verified drug documentation
- **Patient-Aware:** Integrates age, weight, allergies, and conditions into medical reasoning
- **Privacy-Focused:** Local SLM inference ensures medical data never leaves your machine
- **Emergency Detection:** Pre-SLM keyword filtering for life-threatening queries

### Technical Stack

**Backend:**

- FastAPI (async Python web framework)
- LangChain (RAG orchestration)
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- Google Gemma-3-4b (SLM via LM Studio)

**Frontend:**

- React 18 + Vite
- Framer Motion (animations)
- Modern responsive chat UI

**Data Source:**

- OpenFDA API (DailyMed drug labels)
- 50+ FDA-approved medication labels
- Includes contraindications, warnings, dosage guidelines, and drug interactions

### RAG Pipeline

1. **Data Ingestion:** Fetch FDA drug labels via OpenFDA API
2. **Chunking:** RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
3. **Embedding:** all-MiniLM-L6-v2 (384-dim vectors)
4. **Indexing:** ChromaDB with disk persistence
5. **Retrieval:** Top-3 semantic search on user queries
6. **Generation:** Gemma-3-4b with temperature=0.0 for factual responses

## Project Structure

The project is split into two main components:

- `**backend/`**: A Python-based FastAPI server that handles Agentic RAG, data ingestion, evaluation, and SLM integration.
  - `src/`: Core logic, endpoints, RAG agents, and data scraping for medical leaflets.
  - `data/`: Storage for database embeddings and raw/processed document files.
  - `tests/`: End-to-End and unit tests.
- `**frontend/**`: A Vite + React web interface designed to interact seamlessly with the backend API, offering a sleek Chat UX.

## Setup Instructions

### Backend

1. Navigate to the backend directory:
  ```bash
   cd backend
  ```
2. Install Python dependencies:
  ```bash
   pip install -r requirements.txt
  ```
3. Run the development server (make sure to include `src` in your Python path):
  ```bash
   export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
   python src/api/main.py
  ```
   The backend will run on `http://localhost:8000`.

### Frontend

1. Navigate to the frontend directory:
  ```bash
   cd frontend
  ```
2. Install Node.js dependencies:
  ```bash
   npm install
  ```
3. Start the dev server:
  ```bash
   npm run dev
  ```
   The frontend will run on `http://localhost:5173`.

## Key Technical Decisions

### Why ChromaDB?

- Zero-configuration embedded database
- Persistent local storage (no separate server needed)
- Native LangChain integration
- Sufficient performance for 50-1000 document corpus
- Python-native with minimal dependencies

### Why Gemma-3-4b?

- Efficient 4B parameter model runs well on consumer hardware
- Strong instruction-following and factual reasoning capabilities
- Local inference ensures data privacy (HIPAA-friendly)
- Open weights with no API costs
- Temperature 0.0 for deterministic, non-hallucinated responses

### Why all-MiniLM-L6-v2 Embeddings?

- Fast CPU inference (<100ms per query)
- Lightweight (80MB model size)
- Proven semantic similarity performance
- 384-dimensional embeddings balance quality and speed
- Domain-agnostic (works well for medical text without fine-tuning)

### Chunking Strategy

- **Chunk size:** 1000 characters (captures complete medical statements)
- **Overlap:** 200 characters (prevents information loss at boundaries)
- **Separators:** Hierarchical (paragraph → sentence → word)
- **Optimized for:** Precise retrieval of contraindications and dosage guidelines

## Safety Features

- **Pre-SLM Guardrails:** Keyword-based emergency detection (anaphylaxis, overdose, chest pain, etc.)
- **Grounded Generation:** System prompt enforces strict adherence to retrieved FDA documentation
- **No Hallucination Policy:** SLM instructed to say "I don't know" when information is insufficient
- **Patient Context Integration:** All responses consider age, weight, allergies, and conditions
- **Emergency Routing:** Life-threatening queries immediately return emergency guidance

## Testing

Run the test suite:

```bash
cd backend
pytest
```

Tests cover:

- Emergency guardrail triggering
- Patient context formatting
- API endpoint behavior
- Safety mechanism validation

## API Documentation

### Endpoints

#### `POST /api/v1/ask`

Main query endpoint for medical questions.

**Request Body:**

```json
{
  "query": "Is it safe to take ibuprofen for a headache?",
  "patient_context": {
    "age": 45,
    "weight": 80,
    "allergies": ["Penicillin"],
    "conditions": ["Hypertension"]
  }
}
```

**Response:**

```json
{
  "answer": "Based on the retrieved FDA documentation...",
  "status": "success"
}
```

**Status Values:**

- `success`: Normal response with medical information
- `emergency`: Life-threatening query detected, emergency guidance returned

#### `GET /health`

Health check endpoint for monitoring.

**Response:**

```json
{
  "status": "healthy",
  "components": {
    "rag_db": "connected"
  }
}
```

#### `POST /api/v1/feedback`

Records a human feedback event (thumbs up/down, optional suggested correction) for the previous answer. Appended to `backend/data/feedback/feedback.jsonl` — this is the raw material for the DPO preference dataset.

**Request body:**

```json
{
  "query": "my head hurts",
  "patient_context": { "age": 30, "weight": 70, "allergies": [], "conditions": [] },
  "answer": "Take 800 mg ibuprofen every 4 hours.",
  "rating": "down",
  "correction": "Ask how long it's been going on first, then suggest OTC options at labeled doses.",
  "status": "success"
}
```

**Response:** `{ "ok": true, "stored": 42 }` (running count of feedback rows on disk).

#### `GET /api/v1/feedback/export`

Converts the collected thumbs-down feedback into DPO-pair stubs (`{prompt, rejected, chosen|null, meta}`) and writes `backend/data/feedback/feedback_dpo_stub.jsonl`. `training/build_dpo_dataset.py --prompts-from feedback` consumes these.

## Data Pipeline

### Building the Vector Database

If you need to rebuild or update the vector database with fresh FDA data:

1. Fetch drug labels from OpenFDA:

```bash
cd backend
python src/data_ingestion/fetch_openfda.py
```

1. Build the vector database:

```bash
python src/data_ingestion/build_vector_db.py
```

This will:

- Load JSON drug labels from `backend/data/raw/dailymed/`
- Chunk documents using RecursiveCharacterTextSplitter
- Generate embeddings using all-MiniLM-L6-v2
- Store vectors in ChromaDB at `backend/data/processed/chroma_db/`

> **Data note:** the structured tools (`dosage_calculator`, `check_drug_interactions`, `check_allergies`) load section-precise JSON from `backend/data/raw/dailymed/`. A bundle of 386 hand-picked FDA labels lives in `backend/data/raw/dailymed_fda_archive/` — copy them into `backend/data/raw/dailymed/` (or re-run `fetch_openfda.py`) so those tools have data to look up. The ChromaDB store used for RAG retrieval is already populated.

## Fine-tuning & RLHF (Phases 2–3)

The runtime model is **not stock Gemma-3-4b** — it's a QLoRA supervised fine-tune followed by a DPO (Direct Preference Optimization) pass, both run on a free Colab/Kaggle T4. See [`backend/training/README.md`](backend/training/README.md) for the full workflow.

**Phase 2 — SFT.** `backend/training/build_qa_dataset.py` generates a synthetic clinical Q&A dataset (`sft_pairs.jsonl`) — drug-label Q&A grounded in the FDA corpus plus 100+ curated symptom-triage themes — using a remote builder LLM (DeepSeek) with judge validation. `finetune_gemma_lora.ipynb` trains the LoRA adapter and exports `medguard-gemma-3-4b-q4_k_m.gguf`.

**Phase 3 — DPO / RLHF.** `backend/training/build_dpo_dataset.py` builds a preference dataset (`dpo_pairs.jsonl`) from three sources:
- **seeded safety hard-negatives** — deterministic `(chosen, rejected)` pairs templated from the symptom themes, where the rejected answer breaks exactly one safety rule (drops the `[EMERGENCY]` tag, over-triages a routine symptom, invents an overconfident dose, names a prescription-only drug);
- **degraded SFT answers** — the (judge-validated) SFT answer is `chosen`; DeepSeek rewrites it into a subtly worse `rejected`; a preference judge (run with a position-swap consistency check) confirms;
- **human feedback** — thumbs-down events from the UI: the downvoted answer is `rejected`, the user correction (or a regenerated-and-judged answer) is `chosen`; these are weighted up.

`finetune_gemma_dpo.ipynb` continues from the SFT adapter (or base Gemma as a fallback) with `trl.DPOTrainer` (T4-tuned: batch 1 ×8 grad-accum, `lr=5e-6`, `beta=0.1`, 1 epoch, `ref_model=None`) and exports `medguard-gemma-3-4b-dpo-q4_k_m.gguf`. `export_to_lmstudio.py` copies it into LM Studio; point `.env`'s `LOCAL_LLM_MODEL` at it and restart the backend — no application code changes.

## Evaluation

`backend/evaluation/` holds a hand-curated held-out test set (`data/eval_set.jsonl`, ~110 cases across `drug_qa`, `symptom_triage`, `adversarial`, `hallucination`) — intentionally NOT LLM-generated. `run_eval.py` scores each case with:
- **rule checks** — `must_contain` (all), `must_contain_any` (≥1), `must_not_contain` (none), expected triage tier — the deterministic headline metric;
- **DeepEval LLM-judge metrics** — Faithfulness, Answer Relevancy, Hallucination (inverted), and a custom G-Eval "Medical Safety" rubric, judged by DeepSeek (never the model under test);
- **classical NLP** — ROUGE-1/L and BLEU against the reference answer, for cases that ship an `expected_output`.

Compare the base, SFT, and DPO models on the same set and judge:

```bash
cd backend
python evaluation/build_eval_set.py                                   # regenerate eval_set.jsonl
python evaluation/run_eval.py --label base --model-override gemma-3-4b
python evaluation/run_eval.py --label sft  --model-override medguard/medguard-gemma-3-4b
python evaluation/run_eval.py --label dpo  --model-override medguard/medguard-gemma-3-4b-dpo
python evaluation/compare_runs.py --runs base sft dpo --per-case      # -> results/comparison.md
```

(Swap the loaded model in LM Studio between runs to match `--model-override`.) Per-run output lands in `evaluation/results/<label>/{cases.jsonl,summary.json,REPORT.md}`; `comparison.md` tabulates all metrics across runs plus the list of cases that flipped pass↔fail.

## Future Enhancements

### Planned RAG Strategies

- **Query Rewriting:** Improve retrieval for ambiguous queries
- **Multi-Query RAG:** Generate query variations for increased recall
- **Hybrid Search:** Combine semantic + keyword (BM25) search
- **Re-ranking:** Second-stage relevance scoring for retrieved chunks
- **Citation Generation:** Include source drug label references in responses

### Advanced Safety Features

- **NeMo Guardrails:** Replace keyword detection with trained classifier
- **Output Validation:** Post-generation fact-checking
- **Confidence Scores:** Uncertainty quantification for predictions
- **Human-in-the-Loop:** Flag ambiguous cases for physician review

### Scalability

- **Vector DB Migration:** Evaluate Qdrant/Weaviate for larger document corpora
- **Model Optimization:** Quantization and acceleration for faster inference
- **Caching Layer:** Redis integration for frequently asked questions

## Contributing

This project is under active development. For milestone deliverables and progress tracking, see `milestone2.tex`.

## Disclaimer

**IMPORTANT:** MedGuardAI is a research prototype and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

## Project status (vs. the course rubric)

- ✅ Agent architecture with multiple tools (LangGraph + 6 clinical tools)
- ✅ Data ingestion of new datasets (OpenFDA pipeline; second loaders pluggable)
- ✅ RAG (ChromaDB + MMR, all-MiniLM-L6-v2 embeddings, hierarchical chunking)
- ✅ Task-specific fine-tuning (QLoRA SFT on synthetic clinical Q&A) — **the runtime SLM is fine-tuned**
- ✅ RLHF (DPO on a preference dataset: seeded safety negatives + degraded SFT answers + a live human-feedback loop)
- ✅ Toxicity / hallucination handling (input + output guards, groundedness check)
- ✅ Evaluation (held-out hand-curated set, rule checks + DeepEval metrics + classical NLP, base→SFT→DPO comparison)

### Remaining / stretch

- Frontend E2E tests (Playwright).
- Hybrid retrieval (semantic + BM25) and a cross-encoder re-ranker.
- A second ingested dataset (e.g. DrugBank interactions) to broaden "data ingestion".
- Re-fetch / refresh the FDA corpus into `backend/data/raw/dailymed/` (see Data note above).

