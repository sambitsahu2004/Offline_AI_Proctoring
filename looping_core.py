"""
looping_core.py
=================
Framework-agnostic "loop" logic shared by the CrewAI pipeline (crew_tools.py
/ crew_pipeline.py). Deliberately has NO import of crewai — every function
here is plain Python you can unit-test without a live LLM, a running crew,
or Ollama being reachable. crew_tools.py imports these and wraps them as
CrewAI Tools / Task guardrails.

Three loops, one per pipeline component, each targeting a specific
hallucination/error surface:

1. vote_vision_result()      — Component 1 (extraction): a single llava call
   on one frame can misread it. Samples the vision model up to
   `max_samples` times and keeps the majority answer, stopping early once
   `min_consensus` samples agree. Reduces single-call noise before an event
   is ever logged as fact.

2. retrieve_with_coverage_loop() — Component 2 (retrieval): one-shot top-k
   retrieval can miss the policy chunk for an event type the session
   actually has. Iteratively retrieves (reformulating the query toward
   whichever event types are still uncovered) until every event type in
   the session is backed by at least one retrieved policy chunk, or
   `max_iterations` is reached.

3. validate_report_against_events() + build_deterministic_fallback_report()
   — Component 3 (report generation): the LLM is *told* to emit exactly
   one timeline row per real event, in order, but nothing previously
   checked that it complied. validate_report_against_events() diffs the
   LLM's markdown table against the real event JSON (this doubles as a
   CrewAI Task guardrail — see crew_pipeline.py). If the LLM still won't
   comply after every retry, build_deterministic_fallback_report() builds
   a report directly from the event data with no LLM call at all — a
   report is always produced, and it is always correct, even in the worst
   case.
"""

import re
import logging
from collections import Counter

logger = logging.getLogger("looping_core")
if not logger.handlers:
    # Falls back to whatever root logging is configured (e.g. by
    # extractor_agent.py) — this module doesn't own its own log file.
    logger.addHandler(logging.NullHandler())

KNOWN_EVENT_TYPES = {"gaze_deviation", "multiple_faces", "no_face", "phone_detected"}


# ---------------------------------------------------------------------------
# 1. Extraction — self-consistency loop over repeated vision-model samples
# ---------------------------------------------------------------------------
def _bucket_human_count(n):
    """0, 1, or '2+' — buckets so 'saw 2 faces' vs 'saw 3 faces' still count
    as agreement (both mean multiple_faces), while still separating the
    no_face / single-face / multiple-faces cases that matter for events."""
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    return "2+"


def vote_vision_result(call_and_parse_fn, max_samples=5, min_consensus=3):
    """Samples `call_and_parse_fn()` (a zero-arg callable that makes one
    vision-model call + parse and returns a result dict in the same shape
    as extractor_agent._parse_vision_response()) up to max_samples times.

    Stops early once `min_consensus` samples agree on the (human_count
    bucket, gaze category, phone visible) triple — no need to burn the
    full sample budget when the model is already consistent. Otherwise
    takes a majority vote across all max_samples draws.

    Returns (result, meta) where result is a single vision-result dict
    (confidences averaged across the winning/agreeing samples) and meta
    describes how the loop went — samples_taken, consensus_reached,
    agreement_count — so callers can log or surface loop diagnostics.
    """
    samples = []
    key_counts = Counter()

    for i in range(max_samples):
        result = call_and_parse_fn()
        samples.append(result)
        key = (
            _bucket_human_count(result["human_count"]),
            result["primary_person_gaze"],
            result["phone_or_device_visible"],
        )
        key_counts[key] += 1

        top_key, top_count = key_counts.most_common(1)[0]
        if top_count >= min_consensus:
            agreeing = [s for s in samples if (
                _bucket_human_count(s["human_count"]),
                s["primary_person_gaze"],
                s["phone_or_device_visible"],
            ) == top_key]
            merged = _merge_agreeing_samples(agreeing)
            logger.debug(
                f"vote_vision_result: consensus reached after {i + 1}/{max_samples} samples "
                f"({top_count} agreeing on {top_key})"
            )
            return merged, {
                "samples_taken": i + 1,
                "consensus_reached": True,
                "agreement_count": top_count,
            }

    # No early consensus — majority vote across everything we drew.
    top_key, top_count = key_counts.most_common(1)[0]
    agreeing = [s for s in samples if (
        _bucket_human_count(s["human_count"]),
        s["primary_person_gaze"],
        s["phone_or_device_visible"],
    ) == top_key]
    merged = _merge_agreeing_samples(agreeing)
    logger.warning(
        f"vote_vision_result: no early consensus after {max_samples} samples — "
        f"using majority vote ({top_count}/{max_samples} agreeing on {top_key})"
    )
    return merged, {
        "samples_taken": max_samples,
        "consensus_reached": False,
        "agreement_count": top_count,
    }


def _merge_agreeing_samples(agreeing):
    """Averages confidences across the samples that agreed with each other,
    keeps the first sample's human_count/gaze/phone flag (they agree by
    construction) and concatenates any distinct other_notes."""
    first = agreeing[0]
    n = len(agreeing)
    notes = "; ".join(dict.fromkeys(s["other_notes"] for s in agreeing if s["other_notes"]))
    return {
        "human_count": first["human_count"],
        "human_count_confidence": sum(s["human_count_confidence"] for s in agreeing) / n,
        "primary_person_gaze": first["primary_person_gaze"],
        "gaze_confidence": sum(s["gaze_confidence"] for s in agreeing) / n,
        "phone_or_device_visible": first["phone_or_device_visible"],
        "phone_confidence": sum(s["phone_confidence"] for s in agreeing) / n,
        "other_notes": notes,
        "vision_call_ok": all(s.get("vision_call_ok", True) for s in agreeing),
    }


# ---------------------------------------------------------------------------
# 2. Retrieval — corrective loop that iterates until every event type in
#    the session is covered by at least one retrieved policy chunk
# ---------------------------------------------------------------------------
def retrieve_with_coverage_loop(retrieve_fn, required_event_types, max_iterations=5, k=4):
    """retrieve_fn(query_text, k) -> list of (chunk_text, metadata_dict).
    metadata_dict is expected to optionally carry an "event_types" key
    (comma-separated string) as tagged by rag_core.seed_knowledge_base()'s
    header parsing — see the rag_core.py edit in this same changeset.

    Loops: start from the general query, check which required_event_types
    are covered by a chunk's tagged event_types (or, if untagged, a plain
    substring match on the chunk text as a fallback signal); for anything
    still uncovered, issue a follow-up query naming those event types
    explicitly and merge in the new results (deduped). Stops when fully
    covered or max_iterations reached.

    Returns (context_str, meta) — meta has iterations_used, covered (set),
    uncovered (set) so callers can log/report on it.
    """
    required = set(required_event_types)
    if not required:
        return "", {"iterations_used": 0, "covered": set(), "uncovered": set()}

    seen_chunks = {}  # chunk_text -> metadata, dedup key = chunk_text itself
    covered = set()

    base_query = "Exam proctoring policy guidance for: " + ", ".join(sorted(required))
    query = base_query

    for iteration in range(1, max_iterations + 1):
        results = retrieve_fn(query, k)
        for chunk_text, metadata in results:
            seen_chunks[chunk_text] = metadata
            tagged = set()
            if metadata and metadata.get("event_types"):
                tagged = {t.strip() for t in metadata["event_types"].split(",") if t.strip()}
            for etype in required:
                if etype in tagged or etype in chunk_text:
                    covered.add(etype)

        uncovered = required - covered
        logger.debug(
            f"retrieve_with_coverage_loop: iteration {iteration}/{max_iterations} — "
            f"covered={sorted(covered)} uncovered={sorted(uncovered)}"
        )
        if not uncovered:
            break

        # Reformulate: next query names exactly what's still missing.
        query = "Proctoring policy severity rules for event types: " + ", ".join(sorted(uncovered))
    else:
        uncovered = required - covered
        if uncovered:
            logger.warning(
                f"retrieve_with_coverage_loop: gave up after {max_iterations} iterations, "
                f"still uncovered: {sorted(uncovered)}"
            )

    context = "\n\n---\n\n".join(seen_chunks.keys())
    return context, {
        "iterations_used": iteration,
        "covered": covered,
        "uncovered": required - covered,
    }


# ---------------------------------------------------------------------------
# 3. Report generation — validate the LLM's timeline table against the
#    real event JSON, and a zero-LLM-call fallback that's always correct
# ---------------------------------------------------------------------------
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")


def _extract_timeline_event_types(report_markdown):
    """Pulls the event_type token out of each data row of the first
    markdown table found after a 'Timeline' heading. Deliberately tolerant
    of column order/wording differences — it just looks for a KNOWN_EVENT_TYPES
    token (optionally backtick-wrapped) anywhere in each table data row."""
    lines = report_markdown.splitlines()
    found_types = []
    in_timeline_section = False
    in_table = False

    for line in lines:
        if line.strip().startswith("#") and "timeline" in line.lower():
            in_timeline_section = True
            in_table = False
            continue
        if in_timeline_section and line.strip().startswith("#") and "timeline" not in line.lower():
            break  # left the Timeline section

        if not in_timeline_section:
            continue

        m = _TABLE_ROW_RE.match(line)
        if not m:
            continue

        cells = [c.strip().strip("`") for c in m.group(1).split("|")]
        # Skip header row and the |---|---| separator row.
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            in_table = True
            continue
        if not in_table:
            # First row after the section heading, before we've seen a
            # separator row, is the header — skip it too.
            if any(c.lower() in {"timestamp", "event type", "event_type"} for c in cells):
                continue

        row_text = " ".join(cells).lower()
        match = next((et for et in KNOWN_EVENT_TYPES if et in row_text or et.replace("_", " ") in row_text), None)
        if match:
            found_types.append(match)

    return found_types


def validate_report_against_events(report_markdown, session_data):
    """Returns (is_valid, error_message). error_message is None on success,
    or a specific, actionable description of the mismatch — this string is
    exactly what should be fed back to the LLM as corrective context on
    retry (and is what crew_pipeline.py's Task guardrail returns).
    """
    expected = [e["event_type"] for e in session_data.get("events", [])]

    if not expected:
        # Zero-event sessions should never reach the LLM at all (rag_core's
        # generate_report() already short-circuits to a template) — but
        # validate defensively anyway in case this gets called directly.
        if re.search(r"\bno\s+(incidents?|events?|anomalies)\b", report_markdown, re.IGNORECASE):
            return True, None
        return False, (
            "This session had ZERO real events, but the report doesn't clearly state that "
            "no incidents were recorded. Rewrite it to reflect zero events."
        )

    found = _extract_timeline_event_types(report_markdown)

    if len(found) != len(expected):
        return False, (
            f"Timeline table has {len(found)} row(s) but the event log has EXACTLY "
            f"{len(expected)} event(s): {expected}. Your table must have exactly "
            f"{len(expected)} row(s), one per event, in this exact order — no more, no fewer."
        )

    if found != expected:
        return False, (
            f"Timeline table rows are {found} but must match the event log exactly, "
            f"in order: {expected}. Fix the event types and/or their order to match exactly."
        )

    extra_types = set(found) - KNOWN_EVENT_TYPES
    if extra_types:
        return False, (
            f"Timeline table references unknown event type(s) {sorted(extra_types)} that are "
            f"not in the proctoring taxonomy ({sorted(KNOWN_EVENT_TYPES)}). Remove them."
        )

    return True, None


_SEVERITY_RULES = {
    # event_type -> list of (max_duration_seconds_or_None, severity)
    # mirrors knowledge_base/event_definitions.md's thresholds; None = "no upper bound"
    "gaze_deviation": [(5, "Minor"), (10, "Medium"), (None, "High")],
    "no_face": [(3, "Minor"), (10, "Medium"), (None, "High")],
    "multiple_faces": [(None, "High")],  # Critical requires "communicating", not detectable from CV alone
    "phone_detected": [(None, "Critical")],
}


def _classify_severity(event):
    duration_sec = event.get("duration_ms", 0) / 1000.0
    rules = _SEVERITY_RULES.get(event["event_type"], [(None, "Medium")])
    for max_dur, severity in rules:
        if max_dur is None or duration_sec <= max_dur:
            return severity
    return rules[-1][1]


def build_deterministic_fallback_report(session_data):
    """Zero-LLM-call report, assembled directly from the real event JSON
    using the same severity thresholds as knowledge_base/event_definitions.md.
    Guaranteed to have the exactly-correct number of rows in the exactly-
    correct order, because it's built from the data, not generated. This is
    the last resort when the LLM won't pass validate_report_against_events()
    even after every retry — the pipeline should still return SOMETHING
    correct rather than fail outright."""
    events = session_data.get("events", [])

    rows = []
    for e in events:
        severity = _classify_severity(e)
        duration_sec = round(e.get("duration_ms", 0) / 1000.0, 1)
        rows.append(
            f"| {e['timestamp']} | `{e['event_type']}` | {duration_sec}s | {severity} | "
            f"confidence {e.get('confidence', 0):.2f} |"
        )
    table = (
        "| Timestamp | Event Type | Duration | Severity | Notes |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n" + "\n".join(rows)
    ) if rows else "No incidents were recorded for this session."

    severities = [_classify_severity(e) for e in events]
    highest = next((s for s in ("Critical", "High", "Medium", "Minor") if s in severities), "None")

    return f"""# Exam Session Audit Report
*(auto-generated directly from event data — LLM report generation did not pass validation)*

## Metadata
- **Session ID**: {session_data.get("session_id", "Unknown")}
- **Candidate Name**: {session_data.get("candidate_name", "Unknown")}
- **Exam Name**: {session_data.get("exam_name", "Unknown")}
- **Date**: {session_data.get("date", "Unknown")}
- **Total Duration**: {session_data.get("duration_minutes", 0)} minutes

## Executive Summary
{len(events)} event(s) were flagged during this session. Highest severity observed: **{highest}**.
This section is intentionally factual/mechanical (not LLM-written) — see the timeline below for
exact event data.

## Chronological Incident Timeline
{table}

## Policy Analysis & Context
Severity for each event was assigned using the thresholds defined in
knowledge_base/event_definitions.md (duration-based rules per event type).

## Recommended Action
{"Human review recommended — see flagged events above." if events else "No action required."}
"""
