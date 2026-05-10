import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'dailymed')
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'chroma_db')

def load_json_labels():
    """Load JSON files from the raw directory."""
    documents = []
    if not os.path.exists(RAW_DATA_DIR):
        print("Raw data directory not found. Please run fetch_openfda.py first.")
        return documents

    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(RAW_DATA_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Combine major fields into a format the agent can easily interpret
                content_parts = []
                content_parts.append(f"Drug Name: {data.get('drug_name', 'Unknown')}")
                
                if data.get('active_ingredients'):
                    content_parts.append(f"Active Ingredients: {', '.join(data.get('active_ingredients'))}")
                
                # We want to keep severe warnings highly visible
                if data.get('boxed_warning'):
                    content_parts.append(f"BOXED WARNING: {' '.join(data.get('boxed_warning'))}")
                    
                for section in ['indications_and_usage', 'contraindications', 'warnings_and_cautions', 'drug_interactions', 'dosage_and_administration']:
                    if data.get(section):
                        # Format section name nicely
                        friendly_name = section.replace("_", " ").title()
                        content_parts.append(f"{friendly_name}:\n" + "\n".join(data.get(section, [])))
                        
                full_text = "\n\n".join(content_parts)
                
                # Create a LangChain Document with rich metadata
                doc = Document(
                    page_content=full_text,
                    metadata={
                        "source": filename,
                        "drug_name": data.get('drug_name', 'Unknown'),
                        "has_boxed_warning": bool(data.get('boxed_warning', []))
                    }
                )
                documents.append(doc)
    return documents

def build_vector_database():
    """Chunk documents and store them in ChromaDB using a fast huggingface embedding model."""
    print("Loading documents...")
    documents = load_json_labels()
    
    if not documents:
        print("No documents to process.")
        return

    print(f"Loaded {len(documents)} drug labels.")
    
    # Text splitting optimized for factual medical retrieval
    # Smaller chunks ensure we retrieve specific contraindications accurately
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    # Use GPU when available; embedding ~50k chunks of MiniLM-L6-v2 takes ~30
    # min on CPU vs ~1-3 min on a modest CUDA GPU.
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing embedding model (all-MiniLM-L6-v2) on {device}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": False},
    )
    
    print("Building Chroma vector store...")
    os.makedirs(os.path.dirname(VECTOR_DB_DIR), exist_ok=True)

    # Chroma caps batch inserts at 5461 records. With ~100k+ chunks we have to
    # add them in chunks ourselves.
    BATCH = 5000
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR, embedding_function=embeddings
    )
    total = len(chunks)
    for start in range(0, total, BATCH):
        batch = chunks[start:start + BATCH]
        vectorstore.add_documents(batch)
        print(f"  inserted {min(start + BATCH, total)}/{total} chunks")

    print(f"Vector store successfully saved to {VECTOR_DB_DIR}")

if __name__ == "__main__":
    build_vector_database()
