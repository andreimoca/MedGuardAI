import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'chroma_db')

class ClinicalAgent:
    def __init__(self):
        # We define a strict system prompt tailored for Safety-First Medical Advice
        self.system_prompt = """You are MedGuardAI, a highly specialized clinical assistant.
Your singular purpose is to provide accurate medication advice, using the provided context as your PRIMARY source of truth.

CRITICAL INSTRUCTIONS:
1. **Never Hallucinate Dosages:** For matters of dosage, contraindications, and drug interactions, you MUST base your answers entirely on the retrieved documentation provided in the context.
2. **"I Don't Know" Fallback:** If the user asks a highly specific medical question and the retrieved context does not contain the answer, you must state: "I do not have sufficient information in the official documentation to answer this safely. Please consult a healthcare professional." However, you ARE allowed to engage in normal, friendly conversation or provide general, universally known health information if the user is just chatting or asking non-critical questions.
3. **Emergency Trigger:** If the user describes a life-threatening symptom (e.g., severe allergic reaction, crushing chest pain, difficulty breathing, suspected overdose), immediately output: "[EMERGENCY] Please seek emergency medical assistance or call your local emergency number immediately."
4. **Context Constraints:** Always integrate the user's demographic and medical context into your reasoning (Age, Weight, Allergies).

Always act with caution, but remain helpful and conversant."""
        
        # Load the local Vector DB
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            print(f"Failed to load Vector DB: {e}. Ensure build_vector_db.py ran successfully.")
            self.retriever = None
            
        # Initialize the LLM (Using Google Gemma-3-4b via LM Studio local server)
        local_llm_url = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
        model_name = os.environ.get("LOCAL_LLM_MODEL", "gemma-3-4b")
        
        try:
            self.llm = ChatOpenAI(
                base_url=local_llm_url,
                api_key="not-needed",
                model=model_name,
                temperature=0.0
            )
            print(f"✓ LLM initialized: {model_name} at {local_llm_url}")
        except Exception as e:
            print(f"WARNING: Failed to initialize local LLM: {e}")
            self.llm = None

    def format_patient_context(self, context: Dict[str, Any]) -> str:
        """Format the patient's parameters into a strict context string."""
        return f"""
Patient Profile:
- Age: {context.get('age', 'Unknown')}
- Weight: {context.get('weight', 'Unknown')} kg
- Known Allergies: {', '.join(context.get('allergies', ['None']))}
- Pre-existing Conditions: {', '.join(context.get('conditions', ['None']))}
"""

    def check_for_emergency(self, query: str) -> bool:
        """
        A very primitive hard-coded guardrail before LLM invocation.
        In a real scenario, this would use a dedicated lightweight classifier (e.g., NeMo Guardrails).
        """
        emergency_keywords = ['anaphylaxis', 'overdose', 'cannot breathe', 'heart attack', 'suicide', 'chest pain']
        return any(keyword in query.lower() for keyword in emergency_keywords)

    def process_query(self, user_query: str, patient_context: Dict[str, Any]) -> str:
        """Run the core RAG pipeline."""
        if not self.llm or not self.retriever:
            return "System Error: LLM or Vector DB not initialized correctly."

        # Hard guardrail check
        if self.check_for_emergency(user_query):
            return "[EMERGENCY GUARDRAIL TRIPPED] Please seek emergency medical assistance immediately. Do not wait."

        # 1. Retrieve Context
        print("Retrieving relevant medical documents...")
        docs = self.retriever.invoke(user_query)
        retrieved_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        print(f"Retrieved {len(docs)} documents.")

        # 2. Format Prompts
        patient_profile_str = self.format_patient_context(patient_context)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Here is the patient's demographic context:\n{patient_context}\n\nHere is the retrieved medical documentation:\n{retrieved_documents}\n\nUser Question: {query}")
        ])

        chain = prompt_template | self.llm
        
        # 3. Invoke Model
        print("Analyzing query against medical context...")
        response = chain.invoke({
            "patient_context": patient_profile_str,
            "retrieved_documents": retrieved_text,
            "query": user_query
        })

        return response.content

# Example usage if run directly
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = ClinicalAgent()
    
    # Mock context
    context = {
        "age": 45,
        "weight": 80,
        "allergies": ["Penicillin"],
        "conditions": ["Hypertension"]
    }
    
    query = "Is it safe for me to take ibuprofen for a headache?"
    print(f"\nUser Query: {query}")
    print(agent.process_query(query, context))
