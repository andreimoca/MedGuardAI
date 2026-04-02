# MedGuardAI

**Safety-First Medical Assistant powered by Agentic RAG**

## Team Information

**Team Name:** MEDAI
**Team Members:** Andrei Moca, Bogdan Borodi, Calin Pauliuc, Alexandra Petrea, Ana Vaicum

**GitHub Repository:** [https://github.com/andreimoca/MedGuardAI](https://github.com/andreimoca/MedGuardAI)

## Description of the Architecture

MedGuardAI is an AI-driven medical assistant powered by Agentic RAG. It utilizes a FastAPI Python backend to handle retrieving context and communicating with a local Small Language Model (using Google Gemma-3-4b via LM Studio) while maintaining strict guardrails and evaluating patient context (age, weight, conditions, allergies). The frontend is built as a Single Page Application (SPA) using React, Vite, and Framer Motion for a fluid Chat UX. The architecture ensures an absolute focus on safety and clinical accuracy by validating semantic queries against an internal Vector Database built from official drug leaflets.

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

## Next Steps

- Implement and extend Guardrails.
- Add comprehensive E2E tests for the frontend.
- Optimize the embedding retrieval engine.

