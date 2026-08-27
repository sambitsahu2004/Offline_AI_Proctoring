# UPDATED CODE  

"""
streamlit_app.py
==================
Minimal test UI: upload an image/video, get back exactly two things —
the extracted event-log JSON and the generated report. No step-by-step
processing log, no debug panel — all of that now goes to logs/extractor.log
on the backend instead. Every run is persisted to db/proctoring.db.

Place this file inside rag_system/, next to extractor_agent.py, db.py,
and rag_core.py.

RUN:
    streamlit run streamlit_app.py
"""

import os
import json
import base64
import tempfile
import streamlit as st

import db
from extractor_agent import (
    extract_from_image,
    extract_from_video,
    DEFAULT_PHONE_CONF_THRESHOLD,
    DEFAULT_VISION_MODEL,
    VIDEO_SAMPLE_FPS,
)
from rag_core import (
    init_vector_db,
    compile_retrieval_query,
    retrieve_policy_context,
    generate_report,
    generate_report_pdf,
)

st.set_page_config(page_title="Proctoring Pipeline", layout="wide")
db.init_db()

st.title("Offline AI Proctoring")

tab_new, tab_past = st.tabs(["New session", "Past sessions"])


@st.cache_resource(show_spinner="Loading policy database...")
def get_collection():
    return init_vector_db()


with tab_new:
    with st.sidebar:
        st.header("Session details")

        session_id = st.text_input(
            "Session ID",
            value="exam_demo_001",
        )

        candidate_name = st.text_input(
            "Candidate name",
            value="Jane Doe",
        )

        exam_name = st.text_input(
            "Exam name",
            value="CS101 Final",
        )

        st.header("Models")

        vision_model = st.text_input(
            "Ollama vision model (extraction)",
            value=DEFAULT_VISION_MODEL,
            help=(
                "Multimodal model used by Component 1 to inspect frames "
                "(e.g. llava:7b)."
            ),
        )

        ollama_model = st.text_input(
            "Ollama text model (report generation)",
            value="llama3.2:latest",
            help="Text model used to write the final Markdown audit report.",
        )

        st.header("Detection settings")

        phone_threshold = st.slider(
            "Phone/device detection confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_PHONE_CONF_THRESHOLD,
            step=0.05,
            help=(
                "Lower catches less-certain phone/device detections more "
                "readily, at the cost of more false positives."
            ),
        )

        sample_fps = st.slider(
            "Video sample rate (frames/sec sent to vision model)",
            min_value=0.1,
            max_value=2.0,
            value=VIDEO_SAMPLE_FPS,
            step=0.1,
            help=(
                "Video only. Each sampled frame is one Ollama vision call — "
                "keep this low for long clips on CPU."
            ),
        )

    uploaded = st.file_uploader(
        "Upload a photo or video",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    )

    if uploaded is not None:
        suffix = os.path.splitext(uploaded.name)[1].lower()
        is_video = suffix in [".mp4", ".mov", ".avi"]

        # Preserve original filename for filename-based date parsing.
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, uploaded.name)

        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        if is_video:
            st.video(tmp_path)
        else:
            st.image(tmp_path)

        if st.button("Run analysis", type="primary"):
            with st.spinner(
                "Analyzing (this can take a while — each frame is a vision-model call)..."
            ):
                if is_video:
                    session_data = extract_from_video(
                        tmp_path,
                        session_id,
                        candidate_name,
                        exam_name,
                        sample_fps=sample_fps,
                        model_name=vision_model,
                        phone_conf_threshold=phone_threshold,
                    )
                else:
                    session_data = extract_from_image(
                        tmp_path,
                        session_id,
                        candidate_name,
                        exam_name,
                        model_name=vision_model,
                        phone_conf_threshold=phone_threshold,
                    )

                collection = get_collection()
                query = compile_retrieval_query(session_data)
                context = retrieve_policy_context(collection, query, k=4)

                report_md = generate_report(
                    session_data,
                    context,
                    model_name=ollama_model,
                )

                # New: generate PDF bytes for download.
                report_pdf = generate_report_pdf(session_data, report_md)

                model_used = (
                    ollama_model
                    if session_data["events"]
                    else "template (no events)"
                )

                # Existing database storage remains unchanged.
                db.save_session_events(session_data)
                db.save_report(
                    session_id,
                    report_md,
                    model_used=model_used,
                )

            st.subheader("Extracted event log")

            events_for_display = []

            for event in session_data.get("events", []):
                event_display = {
                    key: value
                    for key, value in event.items()
                    if key != "frame_jpeg_b64"
                }
                event_display["has_frame_image"] = bool(
                    event.get("frame_jpeg_b64")
                )
                events_for_display.append(event_display)

            st.json(
                {
                    **{
                        key: value
                        for key, value in session_data.items()
                        if key != "events"
                    },
                    "events": events_for_display,
                }
            )

            frames_present = [
                event
                for event in session_data.get("events", [])
                if event.get("frame_jpeg_b64")
            ]

            if frames_present:
                st.caption(
                    f"{len(frames_present)} flagged frame(s) — stored in the database, not as files:"
                )

                frame_cols = st.columns(min(len(frames_present), 4))

                for i, event in enumerate(frames_present):
                    image_bytes = base64.b64decode(
                        event["frame_jpeg_b64"]
                    )

                    with frame_cols[i % len(frame_cols)]:
                        st.image(
                            image_bytes,
                            caption=(
                                f"{event['event_type']} @ "
                                f"{event['timestamp']}"
                            ),
                        )

            st.subheader("Generated report")
            st.markdown(report_md)

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "Download event log (.json)",
                    data=json.dumps(session_data, indent=2),
                    file_name=f"{session_id}_events.json",
                    mime="application/json",
                )

            with col2:
                st.download_button(
                    "Download report (.pdf)",
                    data=report_pdf,
                    file_name=f"{session_id}_report.pdf",
                    mime="application/pdf",
                )

    else:
        st.info("Upload a file to begin.")


with tab_past:
    st.subheader("Previously analyzed sessions")
    sessions = db.list_sessions()

    if not sessions:
        st.info(
            "No sessions stored yet — run an analysis in the 'New session' tab."
        )

    else:
        labels = [
            f"{session['session_id']} — {session['candidate_name']} "
            f"({session['date']})"
            for session in sessions
        ]

        selected = st.selectbox(
            "Select a session",
            options=range(len(sessions)),
            format_func=lambda i: labels[i],
        )

        chosen_id = sessions[selected]["session_id"]

        stored_session = db.get_session(chosen_id)
        stored_report = db.get_latest_report(chosen_id)

        st.subheader("Stored event log")
        st.json(stored_session)

        stored_frames = db.get_frames_for_session(chosen_id)

        if stored_frames:
            st.caption(
                f"{len(stored_frames)} flagged frame(s) for this session:"
            )

            frame_cols = st.columns(min(len(stored_frames), 4))

            for i, frame in enumerate(stored_frames):
                with frame_cols[i % len(frame_cols)]:
                    st.image(
                        frame["image_bytes"],
                        caption=(
                            f"{frame['event_type']} @ "
                            f"{frame['timestamp']}"
                        ),
                    )

        if stored_report:
            st.subheader(
                f"Stored report (model: {stored_report['model_used']}, "
                f"generated {stored_report['generated_at']})"
            )
            st.markdown(stored_report["report_markdown"])

        else:
            st.info("No report stored for this session yet.")