# rag_core.py
"""Core functions for the offline proctoring RAG system.
This module contains the original logic extracted from the previous
`rag_reporter.py` so that it can be reused by the LangGraph workflow.
"""

import os
import json
import re
import pandas as pd
import chromadb
import ollama

# Define paths (same as before)
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
    """Initializes the ChromaDB client and seeds the knowledge base if empty."""
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=DB_DIR)
    emb_fn = OllamaEmbeddingFunction(model_name="nomic-embed-text")
    collection = client.get_or_create_collection(
        name="proctoring_policy",
        embedding_function=emb_fn,
    )
    if collection.count() == 0:
        print("Vector database is empty. Seeding knowledge base...")
        seed_knowledge_base(collection)
    else:
        print(f"Vector database loaded. Contains {collection.count()} chunks.")
    return collection


def seed_knowledge_base(collection):
    """Chunk and embed markdown documents from the knowledge_base directory."""
    if not os.path.exists(KB_DIR):
        print(f"Error: Knowledge base directory {KB_DIR} not found.")
        return

    doc_id = 0
    for filename in os.listdir(KB_DIR):
        if not filename.lower().endswith('.md'):
            continue
        filepath = os.path.join(KB_DIR, filename)
        print(f"Processing knowledge base file: {filename}")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Split by markdown headers (## or #)
        sections = re.split(r'\n(?=##? )', content)
        chunks = [s.strip() for s in sections if s.strip()]
        documents = []
        metadatas = []
        ids = []
        for i, chunk in enumerate(chunks):
            # Tag this chunk with whichever backtick-wrapped event_type
            # token(s) appear in its header line (e.g. "## Gaze Deviation
            # (`gaze_deviation`)" -> "gaze_deviation"). Used by the CrewAI
            # retrieval loop (looping_core.retrieve_with_coverage_loop) to
            # check whether a session's event types are actually covered
            # by what got retrieved, instead of trusting embedding
            # similarity alone. Purely additive metadata — chunk content
            # and existing retrieve_policy_context() behavior are unchanged.
            header_line = chunk.splitlines()[0] if chunk else ""
            event_type_tags = ",".join(re.findall(r"`([a-z_]+)`", header_line))
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i, "event_types": event_type_tags})
            ids.append(f"doc_{filename.replace('.', '_')}_{doc_id}")
            doc_id += 1
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"Added {len(documents)} chunks from {filename}")


def compile_retrieval_query(session_data):
    """Create a descriptive query based on detected exam events."""
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
    summary_parts = [f"{etype} occurred {cnt} time(s)" for etype, cnt in event_counts.items()]
    summary_str = ", ".join(summary_parts)
    query = f"Exam session issues: {summary_str}. Total duration of flags: {total_duration_ms/1000:.1f} seconds."
    if critical_events:
        query += f" Critical events: {', '.join(critical_events)}."
    return query


def retrieve_policy_chunks(collection, query_text, k=4):
    """Like retrieve_policy_context(), but returns [(chunk_text, metadata), ...]
    instead of one joined string — the CrewAI retrieval loop
    (looping_core.retrieve_with_coverage_loop) needs each chunk's
    event_types metadata to check coverage. Direct (non-CrewAI) pipeline
    code should keep using retrieve_policy_context(); this is additive."""
    results = collection.query(query_texts=[query_text], n_results=k)
    pairs = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = (results.get("metadatas") or [[]])[0] or [{} for _ in docs]
        pairs = list(zip(docs, metas))
    return pairs


def retrieve_policy_context(collection, query_text, k=4):
    """Retrieve top‑k relevant policy chunks from the vector store."""
    print(f"Retrieving top {k} context chunks for query: '{query_text}'")
    results = collection.query(query_texts=[query_text], n_results=k)
    docs = []
    if results and "documents" in results and results["documents"]:
        for doc in results["documents"][0]:
            docs.append(doc)
    return "\n\n---\n\n".join(docs)


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
    """Call Ollama LLM to produce a markdown audit report.

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

    # Prepare events for LLM (convert ms → seconds, percentages). Frame
    # image data (frame_jpeg_b64) is deliberately NOT included here — it's
    # operational data for the frames table, not something the report
    # writer needs to reason about, and including a base64 blob per event
    # would bloat/pollute the prompt for no benefit.
    formatted_events = []
    for ev in events:
        formatted_events.append({
            "timestamp": ev.get("timestamp"),
            "event_type": ev.get("event_type"),
            "duration_seconds": round(ev.get("duration_ms", 0) / 1000.0, 2),
            "confidence_percent": round(ev.get("confidence", 0.0) * 100, 1),
        })
    events_json = json.dumps(formatted_events, indent=2)

    system_prompt = (
        "You are an expert academic integrity proctoring system auditor. Your task is to generate "
        "an objective, factual, and structured report for a human reviewer based ONLY on the provided "
        "EXAM SESSION DATA.\n\n"
        "IMPORTANT RULES:\n"
        "1. Use ONLY the data provided in the 'EXAM SESSION DATA' and 'Event Log JSON' section. "
        "2. Do not declare the student guilty; just present observations and severity.\n"
        "3. Provide a timeline table with timestamps, event types, durations and severity.\n"
        "4. Provide clear recommended actions.\n"
        "5. Output ONLY the markdown report – no extra explanation.\n"
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
Candidate Name: {session_data.get('candidate_name', 'Unknown')}
Exam Name: {session_data.get('exam_name', 'Unknown')}
Session ID: {session_data.get('session_id', 'Unknown')}
Date: {session_data.get('date', 'Unknown')}
Total Duration: {session_data.get('duration_minutes', 0)} minutes

Event Log JSON ({num_events} event(s) total — your table must have exactly {num_events} row(s)):
{events_json}

[INSTRUCTION]
Generate an audit report following this exact structure:
1. Metadata
2. Executive Summary
3. Chronological Incident Timeline (Markdown table) — exactly {num_events} row(s), no more, no fewer
4. Policy Analysis & Context
5. Recommended Action

Remember: Do not add events beyond the {num_events} listed in the Event Log JSON above.
"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            options={"temperature": 0.1},
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return f"Failed to generate report due to Ollama error: {e}"

# End of rag_core.py
