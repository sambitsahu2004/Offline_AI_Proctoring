import os
import sys
import json
from rag_reporter import init_vector_db, compile_retrieval_query, retrieve_policy_context, process_exam_session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_SESSION = os.path.join(BASE_DIR, "data", "sample_session_events.json")

def test_pipeline():
    print("=== STARTING RAG SYSTEM PIPELINE TEST ===")
    
    # 1. Check database and files setup
    print("\n[Step 1] Checking file dependencies...")
    if not os.path.exists(SAMPLE_SESSION):
        print(f"Error: Sample data not found at {SAMPLE_SESSION}")
        sys.exit(1)
    print("Sample session events data: FOUND.")
    
    # 2. Init and Check ChromaDB
    print("\n[Step 2] Initializing vector store...")
    collection = init_vector_db()
    count = collection.count()
    print(f"Vector collection verified. Total chunks stored: {count}")
    if count == 0:
        print("Error: Vector collection is empty. Seeding failed.")
        sys.exit(1)
        
    # 3. Test Retrieval
    print("\n[Step 3] Testing retrieval logic...")
    with open(SAMPLE_SESSION, "r", encoding="utf-8") as f:
        session_data = json.load(f)
        
    query = compile_retrieval_query(session_data)
    print(f"Compiled query text: {query}")
    
    context = retrieve_policy_context(collection, query, k=3)
    print("\n--- RETRIEVED CONTEXT SAMPLES ---")
    lines = context.split("\n")
    # Show first 15 lines of context to verify
    for line in lines[:15]:
        print(f"  {line}")
    print("  ...")
    print("---------------------------------")
    
    # 4. Generate Final Report (uses Ollama)
    print("\n[Step 4] Triggering end-to-end report generation...")
    print("Connecting to local Ollama (Llama 3.2 3B)...")
    
    output_filename = "report_test_output.md"
    process_exam_session(
        session_file=SAMPLE_SESSION,
        model_name="llama3.2:latest",
        output_file=output_filename
    )
    
    output_path = os.path.join(BASE_DIR, output_filename)
    if os.path.exists(output_path):
        print(f"\nSUCCESS: Report file created at: {output_path}")
        print("\n=== TEST RUN COMPLETE ===")
    else:
        print("\nFAILURE: Report file was not created.")
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline()



