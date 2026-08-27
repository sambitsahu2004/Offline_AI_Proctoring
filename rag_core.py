# rag_core.py
"""Core functions for the offline proctoring RAG system.
This module contains the original logic extracted from the previous
`rag_reporter.py` so that it can be reused by the LangGraph workflow.
"""

import os
import io
import json
import re
import html
import pandas as pd
import chromadb
import ollama

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

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
        if not filename.lower().endswith(".md"):
            continue

        filepath = os.path.join(KB_DIR, filename)
        print(f"Processing knowledge base file: {filename}")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        sections = re.split(r"\n(?=##? )", content)
        chunks = [section.strip() for section in sections if section.strip()]

        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            header_line = chunk.splitlines()[0] if chunk else ""
            event_type_tags = ",".join(
                re.findall(r"`([a-z_]+)`", header_line)
            )

            documents.append(chunk)
            metadatas.append(
                {
                    "source": filename,
                    "chunk_index": i,
                    "event_types": event_type_tags,
                }
            )
            ids.append(f"doc_{filename.replace('.', '_')}_{doc_id}")
            doc_id += 1

        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
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
        event_type = event.get("event_type")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

        duration = event.get("duration_ms", 0)
        total_duration_ms += duration

        if event_type == "phone_detected":
            critical_events.append(
                f"phone_detected ({duration / 1000:.1f}s)"
            )

    summary_parts = [
        f"{event_type} occurred {count} time(s)"
        for event_type, count in event_counts.items()
    ]

    summary_str = ", ".join(summary_parts)
    query = (
        f"Exam session issues: {summary_str}. "
        f"Total duration of flags: {total_duration_ms / 1000:.1f} seconds."
    )

    if critical_events:
        query += f" Critical events: {', '.join(critical_events)}."

    return query


def retrieve_policy_chunks(collection, query_text, k=4):
    """Returns policy chunks and metadata for the CrewAI retrieval loop."""
    results = collection.query(query_texts=[query_text], n_results=k)
    pairs = []

    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = (results.get("metadatas") or [[]])[0] or [{} for _ in docs]
        pairs = list(zip(docs, metas))

    return pairs


def retrieve_policy_context(collection, query_text, k=4):
    """Retrieve top-k relevant policy chunks from the vector store."""
    print(f"Retrieving top {k} context chunks for query: '{query_text}'")

    results = collection.query(query_texts=[query_text], n_results=k)
    docs = []

    if results and "documents" in results and results["documents"]:
        for doc in results["documents"][0]:
            docs.append(doc)

    return "\n\n---\n\n".join(docs)


def _build_no_incident_report(session_data):
    """Creates a deterministic report when there are no events."""
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
    """Call Ollama LLM to produce a Markdown audit report."""
    events = session_data.get("events", [])
    num_events = len(events)

    if num_events == 0:
        print("No events in session data — generating template report (no LLM call).")
        return _build_no_incident_report(session_data)

    formatted_events = []

    for event in events:
        formatted_events.append(
            {
                "timestamp": event.get("timestamp"),
                "event_type": event.get("event_type"),
                "duration_seconds": round(
                    event.get("duration_ms", 0) / 1000.0, 2
                ),
                "confidence_percent": round(
                    event.get("confidence", 0.0) * 100, 1
                ),
            }
        )

    events_json = json.dumps(formatted_events, indent=2)

    system_prompt = (
        "You are an expert academic integrity proctoring system auditor. "
        "Your task is to generate an objective, factual, and structured "
        "report for a human reviewer based ONLY on the provided EXAM SESSION DATA.\n\n"
        "IMPORTANT RULES:\n"
        "1. Use ONLY the data provided in the 'EXAM SESSION DATA' and "
        "'Event Log JSON' section.\n"
        "2. Do not declare the student guilty; just present observations "
        "and severity.\n"
        "3. Provide a timeline table with timestamps, event types, durations "
        "and severity.\n"
        "4. Provide clear recommended actions.\n"
        "5. Output ONLY the markdown report – no extra explanation.\n"
        f"6. The Event Log JSON below contains EXACTLY {num_events} event(s). "
        "Your Chronological Incident Timeline table MUST contain EXACTLY "
        f"{num_events} row(s) — one per event in the Event Log JSON, in the "
        "same order. Any events, timestamps, or incidents you see in the "
        "'[RELEVANT POLICIES & SEVERITY REFERENCE]' section below are "
        "formatting references ONLY — they are not real data and must NEVER "
        "be added as rows to the timeline, mentioned in the summary, or "
        "referenced anywhere in your output."
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
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.1},
        )
        return response["message"]["content"]

    except Exception as error:
        print(f"Error communicating with Ollama: {error}")
        return f"Failed to generate report due to Ollama error: {error}"


# PDF severity colours.
SEVERITY_STYLES = {
    "Critical": {"row": "#FDE8E7", "badge": "#C62828", "text": "#FFFFFF"},
    "High": {"row": "#FFF0E0", "badge": "#E65100", "text": "#FFFFFF"},
    "Medium": {"row": "#FFF3E0", "badge": "#F57C00", "text": "#FFFFFF"},
    "Minor": {"row": "#FFFDE7", "badge": "#F9A825", "text": "#1F1F1F"},
}


def get_event_severity(event):
    """
    Deterministic severity mapping used for PDF colour coding.

    Phone detection is always Critical/red. The other classifications
    follow the event definitions already present in the knowledge base.
    """
    event_type = str(event.get("event_type", "")).lower()
    duration_seconds = float(event.get("duration_ms", 0) or 0) / 1000

    if event_type == "phone_detected":
        return "Critical"

    if event_type == "gaze_deviation":
        if duration_seconds >= 10:
            return "High"
        if duration_seconds >= 5:
            return "Medium"
        return "Minor"

    if event_type == "no_face":
        if duration_seconds > 10:
            return "High"
        if duration_seconds >= 3:
            return "Medium"
        return "Minor"

    if event_type == "multiple_faces":
        return "High"

    return "Minor"


def _format_event_name(event_type):
    """Turn phone_detected into Phone Detected."""
    return str(event_type or "unknown").replace("_", " ").title()


def _clean_markdown_text(text):
    """Make a basic Markdown line safe for ReportLab."""
    text = re.sub(r"^\s*[-*]\s+", "• ", text.strip())
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("`", "")
    return html.escape(text)


def _extract_report_sections(report_markdown):
    """Extract Markdown H2 sections from the generated report."""
    sections = []
    current_title = None
    current_lines = []

    for raw_line in report_markdown.splitlines():
        if raw_line.startswith("## "):
            if current_title:
                sections.append((current_title, current_lines))

            current_title = raw_line[3:].strip()
            current_lines = []

        elif not raw_line.startswith("# "):
            current_lines.append(raw_line)

    if current_title:
        sections.append((current_title, current_lines))

    return sections


def _pdf_page_header_footer(canvas, doc):
    """Draw a consistent page header and footer."""
    canvas.saveState()

    canvas.setStrokeColor(HexColor("#D9E2F3"))
    canvas.setLineWidth(0.8)
    canvas.line(
        doc.leftMargin,
        A4[1] - 0.38 * inch,
        A4[0] - doc.rightMargin,
        A4[1] - 0.38 * inch,
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#667085"))
    canvas.drawString(
        doc.leftMargin,
        0.32 * inch,
        "Offline AI Proctoring - Human Review Report",
    )
    canvas.drawRightString(
        A4[0] - doc.rightMargin,
        0.32 * inch,
        f"Page {doc.page}",
    )

    canvas.restoreState()


def generate_report_pdf(session_data, report_markdown):
    """
    Generate a human-readable and colour-coded PDF report.

    The original Markdown report remains stored in SQLite as before.
    This function only creates PDF bytes for the download button.
    """
    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.55 * inch,
        title=f"Proctoring Report - {session_data.get('session_id', 'Unknown')}",
        author="Offline AI Proctoring",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=HexColor("#17365D"),
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=HexColor("#17365D"),
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        spaceAfter=5,
    )

    cell_style = ParagraphStyle(
        "TableCell",
        parent=body_style,
        fontSize=8,
        leading=10,
        spaceAfter=0,
    )

    story = []

    story.append(Paragraph("Exam Session Audit Report", title_style))
    story.append(
        Paragraph(
            "Automatically generated for human review. Flagged events are "
            "observations only and do not determine misconduct.",
            body_style,
        )
    )

    metadata = [
        ["Session ID", str(session_data.get("session_id", "Unknown"))],
        ["Candidate", str(session_data.get("candidate_name", "Unknown"))],
        ["Exam", str(session_data.get("exam_name", "Unknown"))],
        ["Date", str(session_data.get("date", "Unknown"))],
        ["Duration", f"{session_data.get('duration_minutes', 0)} minutes"],
        ["Total flags", str(len(session_data.get("events", [])))],
    ]

    metadata_table = Table(
        metadata,
        colWidths=[1.25 * inch, 5.85 * inch],
    )

    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF1FB")),
                ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#17365D")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.35, HexColor("#C9D6E8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(metadata_table)
    story.append(Paragraph("Chronological Incident Timeline", section_style))

    events = session_data.get("events", [])

    if events:
        table_rows = [[
            "Timestamp",
            "Event type",
            "Duration",
            "Severity",
            "Confidence",
            "Observation",
        ]]

        row_severities = []

        for event in events:
            severity = get_event_severity(event)
            severity_style = SEVERITY_STYLES[severity]
            row_severities.append(severity)

            duration_seconds = float(event.get("duration_ms", 0) or 0) / 1000
            confidence = float(event.get("confidence", 0) or 0) * 100
            event_name = _format_event_name(event.get("event_type"))

            severity_badge_style = ParagraphStyle(
                f"{severity}Badge",
                parent=cell_style,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
                textColor=HexColor(severity_style["text"]),
            )

            table_rows.append(
                [
                    Paragraph(
                        html.escape(str(event.get("timestamp", "Unknown"))),
                        cell_style,
                    ),
                    Paragraph(html.escape(event_name), cell_style),
                    Paragraph(f"{duration_seconds:.1f}s", cell_style),
                    Paragraph(severity, severity_badge_style),
                    Paragraph(f"{confidence:.1f}%", cell_style),
                    Paragraph(
                        html.escape(
                            f"{event_name} detected for {duration_seconds:.1f} seconds."
                        ),
                        cell_style,
                    ),
                ]
            )

        timeline_table = Table(
            table_rows,
            colWidths=[
                0.85 * inch,
                1.25 * inch,
                0.65 * inch,
                0.85 * inch,
                0.75 * inch,
                2.85 * inch,
            ],
            repeatRows=1,
        )

        timeline_style = [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.35, HexColor("#D0D5DD")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]

        for row_number, severity in enumerate(row_severities, start=1):
            severity_style = SEVERITY_STYLES[severity]

            timeline_style.extend(
                [
                    (
                        "BACKGROUND",
                        (0, row_number),
                        (-1, row_number),
                        HexColor(severity_style["row"]),
                    ),
                    (
                        "BACKGROUND",
                        (3, row_number),
                        (3, row_number),
                        HexColor(severity_style["badge"]),
                    ),
                ]
            )

        timeline_table.setStyle(TableStyle(timeline_style))
        story.append(timeline_table)

    else:
        story.append(
            Paragraph(
                "No incidents were recorded for this session.",
                body_style,
            )
        )

    # Include the AI-generated narrative, without duplicating the PDF
    # metadata and colour-coded timeline.
    allowed_sections = {
        "Executive Summary",
        "Policy Analysis & Context",
        "Recommended Action",
    }

    for title, lines in _extract_report_sections(report_markdown):
        if title not in allowed_sections:
            continue

        story.append(Paragraph(title, section_style))

        for line in lines:
            clean_line = line.strip()

            # The PDF creates its own timeline table, so skip Markdown table rows.
            if (
                not clean_line
                or clean_line.startswith("|")
                or clean_line.startswith("| :")
            ):
                continue

            story.append(
                Paragraph(
                    _clean_markdown_text(clean_line),
                    body_style,
                )
            )

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Severity legend: Critical = red, High = dark orange, "
            "Medium = orange, Minor = yellow.",
            ParagraphStyle(
                "Legend",
                parent=body_style,
                fontSize=8,
                textColor=HexColor("#667085"),
            ),
        )
    )

    doc.build(
        story,
        onFirstPage=_pdf_page_header_footer,
        onLaterPages=_pdf_page_header_footer,
    )

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()