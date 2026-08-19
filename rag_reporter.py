import os
import json
import re
import argparse
import pandas as pd
import chromadb
import ollama

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
DB_DIR = os.path.join(BASE_DIR, "db")

class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        response = ollama.embed(model=self.model_name, input=input)
        return response["embeddings"]

def init_vector_db():
    """Initializes the ChromaDB client and embeds/saves knowledge base docs if not already present."""
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Use the Ollama nomic-embed-text model for local embeddings
    emb_fn = OllamaEmbeddingFunction(model_name="nomic-embed-text")

    collection = client.get_or_create_collection(
        name="proctoring_policy",
        embedding_function=emb_fn
    )

    # Check if collection is empty. If so, seed it.
    if collection.count() == 0:
        print("Vector database is empty. Seeding knowledge base...")
        seed_knowledge_base(collection)
    else:
        print(f"Vector database loaded. Contains {collection.count()} chunks.")

    return collection

def seed_knowledge_base(collection):
    """Chunks and embeds markdown documents from the knowledge_base directory."""
    if not os.path.exists(KB_DIR):
        print(f"Error: Knowledge base directory {KB_DIR} not found.")
        return

    doc_id = 0
    for filename in os.listdir(KB_DIR):
        if filename.endswith(".md"):
            filepath = os.path.join(KB_DIR, filename)
            print(f"Processing knowledge base file: {filename}")

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Chunking strategy: split by markdown H2 (##) or H1 (#) headers to keep logical sections intact.
            # If no headers, split by double newlines.
            chunks = []
            sections = re.split(r'\n(?=##? )', content)

            for section in sections:
                section = section.strip()
                if section:
                    chunks.append(section)

            # Embed and add chunks to ChromaDB
            documents = []
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"source": filename, "chunk_index": i})
                ids.append(f"doc_{filename.replace('.', '_')}_{doc_id}")
                doc_id += 1

            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Added {len(documents)} chunks from {filename}")

def compile_retrieval_query(session_data):
    """
    Creates a descriptive query for RAG based on the frequency and severity of detected events.
    """
    events = session_data.get("events", [])
    if not events:
        return "No anomalies detected in the exam session."

    event_counts = {}
    total_duration_ms = 0
    critical_events = []

    for event in events:
        etype = event.get("event_type")
        event_counts[etype] = event_counts.get(etype, 0) + 1
        duration = event.get("duration_ms", 0)
        total_duration_ms += duration

        if etype == "phone_detected":
            critical_events.append(f"phone_detected ({duration/1000:.1f}s)")

    summary_parts = []
    for etype, count in event_counts.items():
        summary_parts.append(f"{etype} occurred {count} time(s)")

    summary_str = ", ".join(summary_parts)
    query = f"Exam session issues: {summary_str}. Total duration of flags: {total_duration_ms/1000:.1f} seconds."

    if critical_events:
        query += f" Critical events: {', '.join(critical_events)}."

    return query

def retrieve_policy_context(collection, query_text, k=4):
    """Retrieves top-k relevant policy/style chunks from the vector store."""
    print(f"Retrieving top {k} context chunks for query: '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=k
    )

    retrieved_docs = []
    if results and 'documents' in results and results['documents']:
        for doc in results['documents'][0]:
            retrieved_docs.append(doc)

    return "\n\n---\n\n".join(retrieved_docs)


def _build_no_incident_report(session_data):
    """Deterministic, template-generated report for sessions with zero real
    events. No LLM call at all — nothing for a model to hallucinate about,
    and it renders instantly."""
    return f"""# Exam Session Audit Report

## Metadata
- **Session ID**: {session_data.get("session_id", "Unknown")}
- **Candidate Name**: {session_data.get("candidate_name", "Unknown")}
- **Exam Name**: {session_data.get("exam_name", "Unknown")}
- **Date**: {session_data.get("date", "Unknown")}
- **Total Duration**: {session_data.get("duration_minutes", 0)} minutes

## Executive Summary
No anomalies were detected during this session. The candidate's face was
consistently present and within normal gaze range, and no prohibited
objects were identified.

## Chronological Incident Timeline
No incidents were recorded for this session.

## Policy Analysis & Context
Not applicable — no events were flagged for policy review.

## Recommended Action
No action required. This session does not require human review.
"""


def generate_report(session_data, context, model_name="llama3.2:latest"):
    """
    Sends the context, event data, and prompts to the local Ollama LLM
    to generate the final structured Markdown report.

    If the session has zero real events, this skips the LLM entirely and
    returns a deterministic template report — this is what prevents the
    model from inventing events it saw in the style-guide examples when
    there's nothing real to report on.
    """
    events = session_data.get("events", [])
    num_events = len(events)

    if num_events == 0:
        print("No events in session data — generating template report (no LLM call).")
        return _build_no_incident_report(session_data)

    print(f"Calling Ollama model '{model_name}' to generate report...")

    # Format events list to convert ms to seconds and make it clean for the LLM.
    # Frame image data (frame_jpeg_b64) is deliberately excluded — it's
    # operational data for the frames table, not something the report
    # writer needs, and would bloat the prompt with a base64 blob per event.
    formatted_events = []
    for event in events:
        formatted_event = {
            "timestamp": event.get("timestamp"),
            "event_type": event.get("event_type"),
            "duration_seconds": round(event.get("duration_ms", 0) / 1000.0, 2),
            "confidence_percent": round(event.get("confidence", 0.0) * 100, 1),
        }
        formatted_events.append(formatted_event)

    events_json = json.dumps(formatted_events, indent=2)

    system_prompt = (
        "You are an expert academic integrity proctoring system auditor. Your task is to generate "
        "an objective, factual, and structured report for a human reviewer based ONLY on the provided EXAM SESSION DATA.\n\n"
        "IMPORTANT RULES:\n"
        "1. Use ONLY the data provided in the 'EXAM SESSION DATA' and 'Event Log JSON' section. Do NOT copy names, session IDs, dates, or events from the style guide examples. The candidate name is John Doe, and the session ID is exam_session_demo_999.\n"
        "2. Do not declare the student 'guilty' of cheating, and do not use terms like 'cheated' or 'guilty'. "
        "Simply state the observations, severity tiers from policy, and details objectively. The final verdict is reserved for a human investigator.\n"
        "3. Provide a timeline table with human-readable timestamps, event types, durations in seconds, and severity classifications.\n"
        "4. Provide clear recommended actions for next steps.\n"
        "5. Output ONLY the markdown report itself. Do NOT output conversational text, introduction, or retrieved context/instruction blocks.\n"
        f"6. The Event Log JSON below contains EXACTLY {num_events} event(s). Your Chronological Incident Timeline "
        f"table MUST contain EXACTLY {num_events} row(s) — one per event in the Event Log JSON, in the same order. "
        "Any events, timestamps, or incidents you see in the '[RELEVANT POLICIES & SEVERITY REFERENCE]' section "
        "below (including any report examples) are formatting references ONLY — they are not real data and must "
        "NEVER be added as rows to the timeline, mentioned in the summary, or referenced anywhere in your output."
    )

    user_prompt = f"""
[RELEVANT POLICIES & SEVERITY REFERENCE]
{context}

[EXAM SESSION DATA TO AUDIT]
Candidate Name: {session_data.get("candidate_name", "Unknown")}
Exam Name: {session_data.get("exam_name", "Unknown")}
Session ID: {session_data.get("session_id", "Unknown")}
Date: {session_data.get("date", "Unknown")}
Total Duration: {session_data.get("duration_minutes", 0)} minutes

Event Log JSON ({num_events} event(s) total — your table must have exactly {num_events} row(s)):
{events_json}

[INSTRUCTION]
Generate an audit report for this specific candidate session based on the event log.
The report must follow this exact structure:
1. Metadata (Session ID, Candidate Name, Exam Name, Date, Duration)
2. Executive Summary (High-level summary of findings for this session)
3. Chronological Incident Timeline (Markdown table containing: Timestamp, Event Type, Duration in seconds, Severity, Description, and Policy Rule Reference) — exactly {num_events} row(s), no more, no fewer
4. Policy Analysis & Context (Explain how each event category maps to the severity defined in the policies)
5. Recommended Action (Factual next steps for a human auditor)

Remember: Output ONLY the markdown report. Do not copy the context or examples. Do not add events beyond the {num_events} listed in the Event Log JSON above.
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.1  # Very low temperature for highly deterministic, factual reports
            }
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return f"Failed to generate report due to Ollama error: {e}"

def process_exam_session(session_file, model_name="llama3.2:latest", output_file=None):
    """Main workflow to load events, run RAG, and write the report."""
    if not os.path.exists(session_file):
        print(f"Error: Session file {session_file} does not exist.")
        return

    with open(session_file, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    # 1. Init Vector DB and retrieve collection
    collection = init_vector_db()

    # 2. Compile query
    query = compile_retrieval_query(session_data)

    # 3. Retrieve context
    context = retrieve_policy_context(collection, query, k=4)

    # 4. Generate report
    report = generate_report(session_data, context, model_name)

    # 5. Output
    if output_file:
        output_path = os.path.join(BASE_DIR, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Success! Report generated and saved to: {output_path}")
    else:
        print("\n--- GENERATED REPORT ---")
        print(report)
        print("-------------------------")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Proctoring RAG Reporter")
    parser.add_argument("--session", type=str, default="data/sample_session_events.json", help="Path to exam session JSON file")
    parser.add_argument("--model", type=str, default="llama3.2:latest", help="Ollama LLM model to use")
    parser.add_argument("--output", type=str, default="report_output.md", help="Filename to output markdown report")

    args = parser.parse_args()

    # Ensure relative paths work correctly
    session_path = os.path.join(BASE_DIR, args.session)
    process_exam_session(session_path, args.model, args.output)
