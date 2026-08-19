"""
crew_pipeline.py
==================
The CrewAI-based version of the pipeline, matching Offline_AI_Proctoring.pdf's
"System Architecture" (Component 1 -> 2 -> 3, sequential CrewAI agents).

This is a SEPARATE, ADDITIVE path alongside run_full_demo.py — it does not
replace it. run_full_demo.py (extractor_agent.py + rag_core.py directly) is
faster and already fully tested; crew_pipeline.py trades speed for the
hallucination-hardening loops this changeset adds:

    Component 1 (Image Extractor Agent)  -> VisionExtractionTool:
        self-consistency loop (looping_core.vote_vision_result)
    Component 2 (RAG Diagnosis Agent)    -> PolicySearchTool:
        corrective-retrieval loop (looping_core.retrieve_with_coverage_loop)
    Component 3 (Report Analysis Agent)  -> ReportBuilderTool:
        generate -> validate -> retry loop, via a CrewAI Task guardrail
        (looping_core.validate_report_against_events), with a deterministic
        zero-LLM-call fallback report if every retry is exhausted
        (looping_core.build_deterministic_fallback_report)

CURRENT SCOPE: single IMAGE sessions only. Video would need a per-sampled-
-frame version of this same crew (one Task per frame, or a tool that loops
internally like extract_from_video() already does) — that's a bigger lift
and video already works end-to-end through run_full_demo.py today, so it's
deliberately left out of this first CrewAI pass. Ask if you want that added
next.

SETUP
-----
    pip install crewai
    ollama pull llava:7b
    ollama pull llama3.1:8b     (or reuse llama3.2:latest — either works,
                                  just pass --text-model to match)

    ⚠ pip flagged real version conflicts installing crewai alongside this
    project's streamlit/chromadb versions (pyarrow, starlette, websockets).
    Nothing broke on import when tested, but a SEPARATE virtual environment
    for the CrewAI path is the safer bet — see requirements-crewai.txt.

USAGE
-----
    python crew_pipeline.py --input data/demo.jpg --session-id exam_001 \
        --candidate "Jane Doe" --exam "CS101 Final" \
        --report-output report_from_crew.md
"""

import os
import json
import argparse
import logging

from crewai import Agent, Task, Crew, Process, LLM

import db
import looping_core as lc
from crew_tools import VisionExtractionTool, PolicySearchTool, ReportBuilderTool

logger = logging.getLogger("crew_pipeline")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
GUARDRAIL_MAX_RETRIES = 5  # per-project choice: prioritize safety over latency


def _make_llm(text_model, base_url):
    return LLM(model=f"ollama/{text_model}", base_url=base_url, temperature=0.1)


def _build_agents(text_model, base_url, vision_model):
    vision_tool = VisionExtractionTool()
    vision_tool.vision_model = vision_model
    policy_tool = PolicySearchTool()
    report_tool = ReportBuilderTool()

    llm = _make_llm(text_model, base_url)

    image_extractor_agent = Agent(
        role="Multimodal Image & Metadata Analyzer",
        goal="Analyze raw candidate exam images and file metadata to extract timestamps and detect "
             "visual anomalies (faces, gaze direction, phones/devices), returning a structured JSON event log.",
        backstory="An expert in computer vision and multimodal image analysis. Inspects files and calls "
                   "the vision tool to return a structured JSON log of events — never invents events beyond "
                   "what the tool reports.",
        tools=[vision_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    rag_diagnosis_agent = Agent(
        role="Proctoring Policy Matcher & Compliance Auditor",
        goal="Perform semantic search on the extracted event JSON to map anomalies to institutional "
             "proctoring guidelines and severity levels.",
        backstory="A detail-oriented compliance officer. Takes the structured JSON findings from the "
                   "extractor, queries the policy collection, and aligns events with specific violation "
                   "classes and severity levels — never guesses a severity that isn't grounded in a "
                   "retrieved policy chunk.",
        tools=[policy_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    report_analysis_agent = Agent(
        role="Post-Exam Summary Writer",
        goal="Synthesize the raw event data and retrieved policy context into a professional, "
             "human-readable audit report.",
        backstory="A technical writer specializing in educational integrity. Translates event logs into "
                   "a narrative summary with a chronological timeline, severity indexes, and recommended "
                   "actions — the timeline must match the real event log exactly, row for row.",
        tools=[report_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return image_extractor_agent, rag_diagnosis_agent, report_analysis_agent


# ---------------------------------------------------------------------------
# Task-level guardrails — these are what turn "generate" into "generate ->
# validate -> retry" using CrewAI's own retry mechanism (Task.guardrail +
# Task.guardrail_max_retries), rather than a hand-rolled while-loop.
# ---------------------------------------------------------------------------
def _extraction_guardrail(task_output):
    """Task 1 guardrail: the agent's final answer must be the tool's raw
    JSON (or trivially wrapped), not commentary — and must have the shape
    generate_report()/db.py expect."""
    text = task_output.raw.strip()
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError) as e:
        return False, f"Your final answer must be ONLY the JSON returned by vision_extraction_tool, unmodified. Parse error: {e}"

    if "events" not in data or not isinstance(data["events"], list):
        return False, "The JSON must have an 'events' list (even if empty) — return the tool's output unchanged."

    bad_types = {e.get("event_type") for e in data["events"]} - lc.KNOWN_EVENT_TYPES
    if bad_types:
        return False, f"Unknown event_type(s) {sorted(bad_types)} — only {sorted(lc.KNOWN_EVENT_TYPES)} are valid. Return the tool's output unchanged, do not invent event types."

    return True, text[start:end]


def _retrieval_guardrail(task_output):
    """Task 2 guardrail: must actually have retrieved something, not just
    passed through an empty/near-empty answer."""
    text = task_output.raw.strip()
    if len(text) < 20:
        return False, "Your final answer is too short to be real policy context — call policy_search_tool and return its full output."
    return True, text


def _make_report_guardrail(extraction_task):
    """Task 3 guardrail: validates the report against the REAL event JSON
    from Task 1's completed output (read at call-time via closure, since
    Task 1 has already run by the time Task 3's guardrail fires in a
    sequential crew)."""
    def _guardrail(task_output):
        try:
            raw = extraction_task.output.raw.strip()
            start, end = raw.index("{"), raw.rindex("}") + 1
            session_data = json.loads(raw[start:end])
        except Exception as e:
            # Can't validate without the real event data — fail closed
            # rather than silently accepting an unverified report.
            return False, f"Could not read the event log from Task 1 to validate against ({e}). Re-emit your report unchanged."

        is_valid, error = lc.validate_report_against_events(task_output.raw, session_data)
        if is_valid:
            return True, task_output.raw
        return False, error

    return _guardrail


def run_crew_session(image_path, session_id, candidate_name, exam_name,
                      vision_model="llava:7b", text_model="llama3.1:8b",
                      base_url=DEFAULT_OLLAMA_BASE_URL,
                      phone_conf_threshold=0.5, guardrail_max_retries=GUARDRAIL_MAX_RETRIES):
    """Runs the 3-agent CrewAI pipeline for one image session. Returns
    (session_data: dict, report_markdown: str, used_fallback: bool)."""
    image_extractor_agent, rag_diagnosis_agent, report_analysis_agent = _build_agents(text_model, base_url, vision_model)

    extraction_task = Task(
        description=(
            f"Call vision_extraction_tool with image_path='{image_path}', session_id='{session_id}', "
            f"candidate_name='{candidate_name}', exam_name='{exam_name}', "
            f"phone_conf_threshold={phone_conf_threshold}. "
            "Return ONLY the tool's JSON output as your final answer — do not add commentary, "
            "do not summarize it in prose, do not invent additional events."
        ),
        expected_output="The exact JSON event log returned by vision_extraction_tool.",
        agent=image_extractor_agent,
        guardrail=_extraction_guardrail,
        guardrail_max_retries=guardrail_max_retries,
    )

    retrieval_task = Task(
        description=(
            "Take the JSON event log from the previous task and call policy_search_tool with it as "
            "session_json. Return the tool's full context output as your final answer, unmodified."
        ),
        expected_output="The retrieved policy context text.",
        agent=rag_diagnosis_agent,
        context=[extraction_task],
        guardrail=_retrieval_guardrail,
        guardrail_max_retries=guardrail_max_retries,
    )

    report_task = Task(
        description=(
            f"Call report_builder_tool with session_json set to the JSON event log from the extraction "
            f"task, policy_context set to the retrieved context from the retrieval task, and "
            f"text_model='{text_model}'. Return the tool's report as your final answer, unmodified. "
            "The report's Chronological Incident Timeline table MUST have exactly one row per event in "
            "the event log, in the same order, with the same event types — no more, no fewer, none invented."
        ),
        expected_output="The final markdown audit report.",
        agent=report_analysis_agent,
        context=[extraction_task, retrieval_task],
        guardrail=_make_report_guardrail(extraction_task),
        guardrail_max_retries=guardrail_max_retries,
    )

    crew = Crew(
        agents=[image_extractor_agent, rag_diagnosis_agent, report_analysis_agent],
        tasks=[extraction_task, retrieval_task, report_task],
        process=Process.sequential,
        verbose=True,
    )

    used_fallback = False
    try:
        crew.kickoff()
        report_markdown = report_task.output.raw
    except Exception as e:
        # Every guardrail retry exhausted somewhere in the chain — this is
        # exactly the case build_deterministic_fallback_report() exists
        # for. We still need real event data to build it from; if even
        # extraction never produced valid JSON, there's nothing to fall
        # back to and we re-raise.
        logger.error(f"Crew pipeline failed after exhausting guardrail retries: {e}")
        raw = extraction_task.output.raw if extraction_task.output else None
        if not raw:
            raise
        start, end = raw.index("{"), raw.rindex("}") + 1
        session_data_fallback = json.loads(raw[start:end])
        report_markdown = lc.build_deterministic_fallback_report(session_data_fallback)
        used_fallback = True
        return session_data_fallback, report_markdown, used_fallback

    session_json_final = extraction_task.output.raw.strip()
    start, end = session_json_final.index("{"), session_json_final.rindex("}") + 1
    session_data = json.loads(session_json_final[start:end])
    return session_data, report_markdown, used_fallback


def main():
    parser = argparse.ArgumentParser(description="CrewAI multi-agent proctoring pipeline (image sessions only)")
    parser.add_argument("--input", required=True, help="Path to a candidate image")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--exam", required=True)
    parser.add_argument("--vision-model", default="llava:7b")
    parser.add_argument("--text-model", default="llama3.1:8b")
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--phone-conf-threshold", type=float, default=0.5)
    parser.add_argument("--guardrail-max-retries", type=int, default=GUARDRAIL_MAX_RETRIES)
    parser.add_argument("--report-output", default="report_from_crew.md")
    args = parser.parse_args()

    session_data, report_markdown, used_fallback = run_crew_session(
        args.input, args.session_id, args.candidate, args.exam,
        vision_model=args.vision_model, text_model=args.text_model, base_url=args.base_url,
        phone_conf_threshold=args.phone_conf_threshold, guardrail_max_retries=args.guardrail_max_retries,
    )

    db.init_db()
    db.save_session_events(session_data)
    db.save_report(args.session_id, report_markdown, model_used=f"{args.text_model} (crew{'+fallback' if used_fallback else ''})")

    output_path = os.path.join(BASE_DIR, args.report_output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    print(f"{'[FALLBACK USED] ' if used_fallback else ''}Report written to {output_path}")


if __name__ == "__main__":
    main()
