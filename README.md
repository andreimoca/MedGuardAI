# Project Name: MedGuardAI

## Description of the Architecture

MedGuardAI is an AI-driven medical assistant powered by Agentic RAG. It utilizes a FastAPI Python backend to handle retrieving context and communicating with LLMs (inference via Groq API) while maintaining strict guardrails and evaluating patient context (age, weight, conditions, allergies). The frontend is built as a Single Page Application (SPA) using React, Vite, and Framer Motion for a fluid Chat UX. The architecture ensures an absolute focus on safety and clinical accuracy by validating semantic queries against an internal Vector Database built from official drug leaflets.

## Project Structure

The project is split into two main components:
- **`backend/`**: A Python-based FastAPI server that handles Agentic RAG, data ingestion, evaluation, and SLM integration.
  - `src/`: Core logic, endpoints, RAG agents, and data scraping for medical leaflets.
  - `data/`: Storage for database embeddings and raw/processed document files.
  - `tests/`: End-to-End and unit tests.
- **`frontend/`**: A Vite + React web interface designed to interact seamlessly with the backend API, offering a sleek Chat UX.

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

## Next Steps
- Implement and extend Guardrails.
- Add comprehensive E2E tests for the frontend.
- Optimize the embedding retrieval engine.
