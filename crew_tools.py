"""
crew_tools.py
===============
CrewAI Tool wrappers around the existing extractor_agent.py / rag_core.py
functions, each one running its corresponding loop from looping_core.py
internally. These are the Tools handed to the three Agents in
crew_pipeline.py.

Design choice: the LOOPS themselves (consensus voting, corrective
retrieval, report validation) are plain deterministic Python in
looping_core.py, not something delegated to an agent's own LLM reasoning.
An agent's LLM decides *when* to call a tool and *what to do with the
result*; it should not be relied on to decide "did my vision model sample
actually converge" — that's a mechanical check these tools make for it.
This is what makes the hallucination guarantees hold regardless of which
Ollama model ends up driving the agents.

Place this file inside rag_system/, next to extractor_agent.py, rag_core.py,
db.py, and looping_core.py.
"""

import json
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

import extractor_agent as ea
import rag_core
import looping_core as lc


# ---------------------------------------------------------------------------
# Component 1 — Image Extractor Agent's tool
# ---------------------------------------------------------------------------
class VisionExtractionArgs(BaseModel):
    image_path: str = Field(..., description="Path to the candidate image file to inspect")
    session_id: str = Field(..., description="Session identifier for this exam session")
    candidate_name: str = Field(..., description="Candidate's name")
    exam_name: str = Field(..., description="Exam name")
    phone_conf_threshold: float = Field(0.5, description="Minimum confidence before a phone/device counts as an event")
    max_samples: int = Field(5, description="Max vision-model samples per frame for the self-consistency loop")
    min_consensus: int = Field(3, description="Samples that must agree before stopping early")


class VisionExtractionTool(BaseTool):
    name: str = "vision_extraction_tool"
    description: str = (
        "Inspects a candidate exam image with a local Ollama multimodal vision model "
        "(llava) and returns a structured JSON event log (same schema as "
        "data/sample_session_events.json). Internally samples the vision model up to "
        "max_samples times per image and only accepts a detection once min_consensus "
        "samples agree — this is a self-consistency loop that filters out one-off "
        "misreads before anything is logged as a real event. Use this exactly once per "
        "image, then pass its JSON output forward unchanged."
    )
    args_schema: Type[BaseModel] = VisionExtractionArgs
    vision_model: str = "llava:7b"

    def _run(self, image_path: str, session_id: str, candidate_name: str, exam_name: str,
              phone_conf_threshold: float = 0.5, max_samples: int = 5, min_consensus: int = 3) -> str:
        import cv2
        frame = cv2.imread(image_path)
        if frame is None:
            return json.dumps({"error": f"Could not read image: {image_path}"})

        capture_dt = ea._extract_image_datetime(image_path)

        def sample_once():
            raw = ea._call_vision_model(frame, self.vision_model)
            return ea._parse_vision_response(raw) if raw is not None else ea._fallback_vision_result("Ollama call failed")

        vision_result, vote_meta = lc.vote_vision_result(sample_once, max_samples=max_samples, min_consensus=min_consensus)
        active_events = ea._classify_frame(vision_result, phone_conf_threshold)

        timestamp_iso = capture_dt.strftime("%Y-%m-%dT%H:%M:%S")
        events = []
        for etype, conf in active_events.items():
            frame_b64 = ea._encode_frame(frame)
            events.append(ea._make_event(session_id, timestamp_iso, etype, conf, 0, frame_b64))

        session_data = {
            "session_id": session_id,
            "candidate_name": candidate_name,
            "exam_name": exam_name,
            "date": capture_dt.strftime("%Y-%m-%d"),
            "duration_minutes": 0,
            "events": events,
            "_extraction_loop_meta": vote_meta,  # diagnostics only — strip before persisting if you don't want this in db.py
        }
        return json.dumps(session_data)


# ---------------------------------------------------------------------------
# Component 2 — RAG Diagnosis Agent's tool
# ---------------------------------------------------------------------------
class PolicySearchArgs(BaseModel):
    session_json: str = Field(..., description="The JSON event log produced by the extraction step")
    max_iterations: int = Field(5, description="Max corrective-retrieval iterations")
    k: int = Field(4, description="Chunks to retrieve per iteration")


class PolicySearchTool(BaseTool):
    name: str = "policy_search_tool"
    description: str = (
        "Given a session's JSON event log, retrieves the relevant proctoring policy "
        "chunks from ChromaDB. Internally loops: retrieves, checks whether every event "
        "type present in the session is actually covered by a retrieved chunk, and if "
        "not, reformulates the query toward exactly what's missing and retrieves again "
        "(up to max_iterations). This is a corrective-retrieval loop that prevents the "
        "report-writer from working off generic or incomplete policy context. Returns "
        "the combined policy context text."
    )
    args_schema: Type[BaseModel] = PolicySearchArgs

    def _run(self, session_json: str, max_iterations: int = 5, k: int = 4) -> str:
        session_data = json.loads(session_json)
        collection = rag_core.init_vector_db()

        required_event_types = {e["event_type"] for e in session_data.get("events", [])}
        if not required_event_types:
            return "No anomalies detected in the exam session — no policy lookup needed."

        def retrieve_fn(query_text, k_):
            return rag_core.retrieve_policy_chunks(collection, query_text, k_)

        context, meta = lc.retrieve_with_coverage_loop(retrieve_fn, required_event_types, max_iterations=max_iterations, k=k)
        if meta["uncovered"]:
            context += (
                f"\n\n[RETRIEVAL WARNING: no policy chunk found for event type(s) "
                f"{sorted(meta['uncovered'])} after {meta['iterations_used']} attempts. "
                f"Flag these for manual policy review rather than guessing a severity.]"
            )
        return context


# ---------------------------------------------------------------------------
# Component 3 — Report Analysis Agent's tool
# ---------------------------------------------------------------------------
class ReportBuilderArgs(BaseModel):
    session_json: str = Field(..., description="The JSON event log produced by the extraction step")
    policy_context: str = Field(..., description="Retrieved policy context text from the policy_search_tool")
    text_model: str = Field("llama3.2:latest", description="Ollama text model for report writing")


class ReportBuilderTool(BaseTool):
    name: str = "report_builder_tool"
    description: str = (
        "Writes the final markdown audit report from a session's event log and the "
        "retrieved policy context. NOTE: this tool does NOT itself validate the report "
        "against the event log — that happens at the Task level via a guardrail (see "
        "crew_pipeline.py), which will automatically re-invoke you with corrective "
        "feedback if your report's timeline table doesn't exactly match the real events. "
        "Just call this once per session and return its output as your final answer."
    )
    args_schema: Type[BaseModel] = ReportBuilderArgs

    def _run(self, session_json: str, policy_context: str, text_model: str = "llama3.2:latest") -> str:
        session_data = json.loads(session_json)
        return rag_core.generate_report(session_data, policy_context, model_name=text_model)
