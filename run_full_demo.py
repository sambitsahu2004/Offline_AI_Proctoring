"""
run_full_demo.py
==================
Ties extractor_agent.py and rag_reporter.py together into one command:
pick an image (or video) -> get the same kind of markdown report as
report_test_output.md.

Place this file inside rag_system/, next to extractor_agent.py and
rag_reporter.py.

USAGE
-----
    python run_full_demo.py --input data/demo.jpg --type image \
        --session-id exam_demo_001 --candidate "Jane Doe" \
        --exam "CS101 Final" --report-output report_from_demo.md

    python run_full_demo.py --input data/demo.mp4 --type video \
        --session-id exam_demo_002 --candidate "Jane Doe" \
        --exam "CS101 Final" --report-output report_from_demo.md \
        --sample-fps 0.5
"""

import os
import argparse

from extractor_agent import extract_from_image, extract_from_video, DEFAULT_PHONE_CONF_THRESHOLD, DEFAULT_VISION_MODEL, VIDEO_SAMPLE_FPS
from rag_reporter import process_exam_session
import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Image/Video -> event log -> RAG report, in one shot")
    parser.add_argument("--input", required=True, help="Path to image or video file")
    parser.add_argument("--type", required=True, choices=["image", "video"])
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--exam", required=True)
    parser.add_argument("--model", default="llama3.2:latest", help="Ollama TEXT model for report generation")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help="Ollama VISION model for extraction (default: llava:7b)")
    parser.add_argument(
        "--session-output", default="data/session_from_demo.json",
        help="Where to save the intermediate extracted event log"
    )
    parser.add_argument(
        "--report-output", default="report_from_demo.md",
        help="Where to save the final markdown report"
    )
    parser.add_argument(
        "--phone-conf-threshold", type=float, default=DEFAULT_PHONE_CONF_THRESHOLD,
        help="Minimum vision-model-reported confidence before a phone/device counts as an event"
    )
    parser.add_argument(
        "--sample-fps", type=float, default=VIDEO_SAMPLE_FPS,
        help="Video only — frames/sec sent to the vision model"
    )
    args = parser.parse_args()

    print(f"[1/3] Extracting events from {args.type}: {args.input} (vision model: {args.vision_model})")
    if args.type == "image":
        session_data = extract_from_image(
            args.input, args.session_id, args.candidate, args.exam,
            model_name=args.vision_model, phone_conf_threshold=args.phone_conf_threshold,
        )
    else:
        session_data = extract_from_video(
            args.input, args.session_id, args.candidate, args.exam,
            sample_fps=args.sample_fps, model_name=args.vision_model,
            phone_conf_threshold=args.phone_conf_threshold,
        )

    session_path = os.path.join(BASE_DIR, args.session_output)
    os.makedirs(os.path.dirname(session_path), exist_ok=True)
    import json
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)
    print(f"      -> {len(session_data['events'])} event(s) found, saved to {session_path}")

    print("[2/3] Persisting events to db/proctoring.db...")
    db.init_db()
    db.save_session_events(session_data)

    print(f"[3/3] Generating RAG report via Ollama ({args.model})...")
    report = process_exam_session(
        session_file=session_path,
        model_name=args.model,
        output_file=args.report_output,
    )
    model_used = args.model if session_data["events"] else "template (no events)"
    db.save_report(args.session_id, report, model_used=model_used)


if __name__ == "__main__":
    main()
