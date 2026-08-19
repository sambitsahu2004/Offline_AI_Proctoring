"""
extractor_agent.py
====================
Component 1 of the pipeline (per Offline_AI_Proctoring.pdf): Image/Video
Information Extractor Agent.

Image / Video -> structured exam event log, in the SAME schema as
data/sample_session_events.json, so its output can be fed directly into
rag_core.py / rag_reporter.py's process_exam_session() and db.py with no
changes to those files.

This version uses a LOCAL OLLAMA MULTIMODAL VISION LLM (default: llava:7b)
to inspect each frame, per the project spec ("Tools: Local Metadata Parser,
Ollama Multimodal Vision Tool"). It REPLACES the earlier MediaPipe Face
Landmarker + YOLOv8-ONNX heuristic pipeline — no face_landmarker.task or
yolov8n.onnx model files are required anymore, only a pulled Ollama vision
model (`ollama pull llava:7b`).

Detections covered (matches event_definitions.md's 4 categories):
    - human_count (from the vision model)  -> no_face / multiple_faces
    - primary_person_gaze (from the vision model) -> gaze_deviation
    - phone_or_device_visible (from the vision model) -> phone_detected

Date/time extraction is unchanged and is NOT vision-based — it stays a
file-metadata step (EXIF -> filename pattern -> file mtime), since that is
far more reliable than asking a vision model to read a clock in-frame.

Place this file inside rag_system/ (next to rag_reporter.py, rag_core.py,
db.py). Requires `ollama` running locally with the chosen model pulled:
    ollama pull llava:7b

USAGE
-----
Single image:
    python extractor_agent.py --input data/demo.jpg --type image \
        --session-id exam_img_001 --candidate "Jane Doe" \
        --exam "CS101 Final" --output data/session_from_image.json

Video:
    python extractor_agent.py --input data/demo.mp4 --type video \
        --session-id exam_vid_001 --candidate "Jane Doe" \
        --exam "CS101 Final" --output data/session_from_video.json \
        --sample-fps 0.5

Then hand the output straight to the existing pipeline:
    python rag_reporter.py --session data/session_from_image.json \
        --output report_from_image.md

(run_full_demo.py in this same folder does both steps in one command.)

NOTE ON PERFORMANCE: a vision-LLM call per frame is much slower than the
old MediaPipe/YOLO pass (seconds per frame on CPU, not milliseconds). For
video, keep --sample-fps low (0.2-1.0) unless running on a GPU-backed
Ollama instance.
"""

import os
import re
import cv2
import json
import base64
import logging
import argparse
import ollama
from datetime import datetime, timedelta
from PIL import Image, ExifTags

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# --------------------------------------------------------------------------
# Backend logging — this replaces all the print()/UI debug output.
# Everything diagnostic (raw model output, why an event did/didn't fire,
# fallback triggers) goes here, not into the JSON output and not into any
# UI. The JSON returned by extract_from_image/video is exactly the
# event-log schema and nothing else.
# --------------------------------------------------------------------------
os.makedirs(LOGS_DIR, exist_ok=True)
logger = logging.getLogger("extractor_agent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "extractor.log"), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

DEFAULT_VISION_MODEL = "llava:7b"

# Minimum vision-model-reported confidence before a phone/device detection
# counts as an event at all. Lower = catches less-certain detections more
# readily, at the cost of more false positives. Tune with --phone-conf-threshold.
DEFAULT_PHONE_CONF_THRESHOLD = 0.5

# Minimum sustained duration (seconds) before a video event is logged at all.
# This is the "event debouncing" step the roadmap calls out explicitly.
MIN_EVENT_DURATION_SEC = {
    "gaze_deviation": 1.5,
    "no_face": 1.0,
    "multiple_faces": 1.0,
    "phone_detected": 0.5,
}

# Analyze 0.5 frames/sec of video by default (one frame every 2s) — a vision
# LLM call is far slower than a heuristic CV pass, so this is intentionally
# much lower than a MediaPipe/YOLO-based extractor would use. Raise it for
# short demo clips if you can afford the extra Ollama calls.
VIDEO_SAMPLE_FPS = 0.5

_GAZE_VALUES = {"on_screen", "looking_away", "looking_down", "unknown"}

VISION_PROMPT = """You are a strict, factual exam-proctoring vision analyst. Look ONLY at what is visible in this single image and answer with STRICT JSON — no prose, no markdown fences, no explanation before or after the JSON object.

Respond with EXACTLY this JSON schema (all fields required):
{
  "human_count": <integer, number of distinct human faces/heads visible in the frame>,
  "human_count_confidence": <float 0.0-1.0, how certain you are of the count>,
  "primary_person_gaze": <one of: "on_screen", "looking_away", "looking_down", "unknown">,
  "gaze_confidence": <float 0.0-1.0>,
  "phone_or_device_visible": <true or false — is a mobile phone or handheld electronic device visible in the person's hand or immediate reach>,
  "phone_confidence": <float 0.0-1.0>,
  "other_notes": "<short factual string, e.g. 'open notebook visible' or 'second monitor visible', or empty string if nothing notable>"
}

Rules:
- Base every field ONLY on what is visibly present in this exact image — do not guess or assume context you cannot see.
- If no human face is visible at all, set human_count to 0 and primary_person_gaze to "unknown".
- If more than one human face is visible, set human_count accordingly regardless of gaze.
- "looking_away" means the head/eyes are turned well off the screen (roughly sideways or beyond); "looking_down" means gaze directed sharply downward (e.g. at a phone, notes, or lap); "on_screen" means facing the camera/screen normally.
- Use LOWER confidence values when the image is blurry, dark, or ambiguous.
- Output ONLY the JSON object. Nothing else.
"""


# --------------------------------------------------------------------------
# Ollama vision call
# --------------------------------------------------------------------------
def _call_vision_model(frame_bgr, model_name):
    """Sends one frame to the Ollama multimodal vision model. Returns the
    raw text content of the model's reply, or None if the call itself
    failed (model not pulled, Ollama not running, etc.) — parsing failures
    are handled separately in _parse_vision_response."""
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        logger.error("Could not JPEG-encode frame for the vision model.")
        return None
    image_bytes = buf.tobytes()

    message = {"role": "user", "content": VISION_PROMPT, "images": [image_bytes]}
    try:
        response = ollama.chat(
            model=model_name,
            messages=[message],
            options={"temperature": 0.0},
            format="json",
        )
    except TypeError:
        # Older ollama-python versions may not accept format= at all.
        try:
            response = ollama.chat(model=model_name, messages=[message], options={"temperature": 0.0})
        except Exception as e:
            logger.error(f"Ollama vision call failed (model={model_name}): {e}")
            return None
    except Exception as e:
        logger.error(f"Ollama vision call failed (model={model_name}): {e}")
        return None

    return response.get("message", {}).get("content", "")


def _fallback_vision_result(reason):
    """A neutral result used when the model call or its output can't be
    trusted — deliberately reports NO events rather than risking a false
    positive flag (e.g. spurious 'no_face') from a broken call."""
    logger.warning(f"Vision result unavailable ({reason}) — treating frame as neutral (no events fired).")
    return {
        "human_count": 1,
        "human_count_confidence": 0.0,
        "primary_person_gaze": "unknown",
        "gaze_confidence": 0.0,
        "phone_or_device_visible": False,
        "phone_confidence": 0.0,
        "other_notes": "",
        "vision_call_ok": False,
    }


def _parse_vision_response(raw_text):
    """Defensively parses the model's JSON reply. Vision LLMs sometimes
    wrap JSON in markdown fences or add stray text despite instructions —
    this strips fences and extracts the first {...} block before parsing."""
    if not raw_text:
        return _fallback_vision_result("empty response from model")

    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return _fallback_vision_result(f"no JSON object found in reply: {text[:150]!r}")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return _fallback_vision_result(f"JSON parse error ({e}) in reply: {text[:150]!r}")

    try:
        human_count = max(0, int(data.get("human_count", 1)))
    except (TypeError, ValueError):
        human_count = 1

    gaze = data.get("primary_person_gaze", "unknown")
    if gaze not in _GAZE_VALUES:
        gaze = "unknown"

    def _clamp01(value, default):
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    return {
        "human_count": human_count,
        "human_count_confidence": _clamp01(data.get("human_count_confidence"), 0.6),
        "primary_person_gaze": gaze,
        "gaze_confidence": _clamp01(data.get("gaze_confidence"), 0.6),
        "phone_or_device_visible": bool(data.get("phone_or_device_visible", False)),
        "phone_confidence": _clamp01(data.get("phone_confidence"), 0.6),
        "other_notes": str(data.get("other_notes", ""))[:300],
        "vision_call_ok": True,
    }


def _classify_frame(vision_result, phone_conf_threshold):
    """Turns one parsed vision-model result into a set of active
    event_types for this frame — same shape/semantics as the previous
    MediaPipe/YOLO-based classifier, so the rest of the pipeline
    (debouncing, event schema) doesn't need to change."""
    active = {}  # event_type -> confidence

    human_count = vision_result["human_count"]
    if human_count == 0:
        active["no_face"] = vision_result["human_count_confidence"]
    elif human_count > 1:
        active["multiple_faces"] = vision_result["human_count_confidence"]
    else:
        gaze = vision_result["primary_person_gaze"]
        if gaze in ("looking_away", "looking_down"):
            active["gaze_deviation"] = vision_result["gaze_confidence"]

    if vision_result["phone_or_device_visible"] and vision_result["phone_confidence"] >= phone_conf_threshold:
        active["phone_detected"] = vision_result["phone_confidence"]

    return active


# --------------------------------------------------------------------------
# File-metadata date/time extraction (unchanged — not vision-based)
# --------------------------------------------------------------------------

# Filename patterns for common camera/sharing apps, used as a fallback when
# EXIF is missing or unreliable — this is often more trustworthy than file
# mtime, since mtime resets to "when I received this" once a file has been
# copied, uploaded, or shared, while the filename usually still encodes the
# original capture time.
_FILENAME_PATTERNS = [
    # Windows Camera app: WIN_20260731_10_02_24_Pro.jpg -> full date + time
    ("win_camera", re.compile(r"WIN_(\d{4})(\d{2})(\d{2})_(\d{2})_(\d{2})_(\d{2})")),
    # WhatsApp: IMG-20260813-WA0001.jpg / VID-20260813-WA0001.mp4 -> date only
    ("whatsapp", re.compile(r"(?:IMG|VID)-(\d{4})(\d{2})(\d{2})-WA\d+")),
    # Generic camera exports: 20260813_153045.jpg -> full date + time
    ("generic", re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?!\d)")),
]


def _parse_datetime_from_filename(filepath):
    """Tries each known filename pattern in turn. Returns a datetime, or
    None if nothing matched. Year is sanity-checked (2000-2100) so this
    doesn't misfire on unrelated digit strings in a filename."""
    filename = os.path.basename(filepath)

    for pattern_name, pattern in _FILENAME_PATTERNS:
        m = pattern.search(filename)
        if not m:
            continue
        groups = [int(g) for g in m.groups()]
        year = groups[0]
        if year < 2000 or year > 2100:
            continue
        try:
            if pattern_name == "whatsapp":
                # date only, no time encoded — defaults to midnight
                year, month, day = groups
                dt = datetime(year, month, day)
                logger.debug(f"Parsed WhatsApp date from filename '{filename}': {dt} (time unknown, defaulted to 00:00:00)")
            else:
                year, month, day, hour, minute, second = groups
                dt = datetime(year, month, day, hour, minute, second)
                logger.debug(f"Parsed {pattern_name} timestamp from filename '{filename}': {dt}")
            return dt
        except ValueError:
            continue  # e.g. impossible date like month=13, try next pattern

    return None


def _extract_image_datetime(image_path):
    """Priority: EXIF (camera's own record) -> filename pattern -> file
    mtime -> now. This is what makes the 'date' field reflect when the
    photo was actually taken, not when the script happened to run or when
    the file was last copied/shared."""
    try:
        img = Image.open(image_path)
        exif = img.getexif()
        if exif:
            tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            for tag in ("DateTimeOriginal", "DateTime"):
                value = tag_map.get(tag)
                if value:
                    try:
                        dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                        logger.debug(f"EXIF {tag} found for {image_path}: {dt}")
                        return dt
                    except ValueError:
                        logger.debug(f"EXIF {tag} present but unparsable ('{value}') for {image_path}")
    except Exception as e:
        logger.debug(f"Could not read EXIF for {image_path}: {e}")

    filename_dt = _parse_datetime_from_filename(image_path)
    if filename_dt:
        return filename_dt

    try:
        mtime = os.path.getmtime(image_path)
        dt = datetime.fromtimestamp(mtime)
        logger.debug(f"No usable EXIF/filename timestamp for {image_path}, using file mtime: {dt}")
        return dt
    except Exception as e:
        logger.warning(f"Could not read file mtime for {image_path}: {e}. Using current time.")
        return datetime.now()


def _extract_video_start_datetime(video_path):
    """Priority: filename pattern -> file mtime -> now. Videos rarely carry
    reliable capture-time metadata in a simple, cross-format way, so the
    filename (when parseable) is preferred over mtime for the same reason
    as images — mtime drifts once a file is shared/copied."""
    filename_dt = _parse_datetime_from_filename(video_path)
    if filename_dt:
        return filename_dt
    try:
        mtime = os.path.getmtime(video_path)
        return datetime.fromtimestamp(mtime)
    except Exception as e:
        logger.warning(f"Could not read file mtime for {video_path}: {e}. Using current time.")
        return datetime.now()


def _encode_frame(frame_bgr):
    """Encodes a frame straight to base64 JPEG bytes — no disk write at
    all. Frames are persisted by db.save_session_events() (as a BLOB in
    the `frames` table, joined to candidate info via session_id), not as
    loose .jpg files. The base64 string travels inside the event dict so
    the JSON event log stays self-contained and easy to hand off, but
    display code (streamlit_app.py) should decode + show it as an image,
    never dump it raw as JSON text."""
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        logger.error("Could not JPEG-encode frame.")
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _make_event(session_id, timestamp_iso, event_type, confidence, duration_ms, frame_jpeg_b64):
    return {
        "session_id": session_id,
        "timestamp": timestamp_iso,
        "event_type": event_type,
        "confidence": round(float(confidence), 3),
        "duration_ms": int(duration_ms),
        "frame_jpeg_b64": frame_jpeg_b64,
    }


# --------------------------------------------------------------------------
# Image mode
# --------------------------------------------------------------------------
def extract_from_image(image_path, session_id, candidate_name, exam_name,
                        model_name=DEFAULT_VISION_MODEL, phone_conf_threshold=DEFAULT_PHONE_CONF_THRESHOLD):
    """A still image has no duration to measure, so each detected condition is logged
    as a single instantaneous event (duration_ms = 0) rather than debounced.

    Returns ONLY the event-log schema (session_id, candidate_name, exam_name,
    date, duration_minutes, events[]) — no debug/UI fields. All diagnostics
    go to logs/extractor.log instead.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    capture_dt = _extract_image_datetime(image_path)

    raw_response = _call_vision_model(frame, model_name)
    vision_result = _parse_vision_response(raw_response) if raw_response is not None else _fallback_vision_result("Ollama call failed")

    active_events = _classify_frame(vision_result, phone_conf_threshold)

    logger.debug(
        f"{image_path}: model={model_name} human_count={vision_result['human_count']} "
        f"gaze={vision_result['primary_person_gaze']} phone_visible={vision_result['phone_or_device_visible']} "
        f"phone_conf={vision_result['phone_confidence']:.3f} (threshold={phone_conf_threshold}) "
        f"notes={vision_result['other_notes']!r} -> events={list(active_events.keys())}"
    )

    timestamp_iso = capture_dt.strftime("%Y-%m-%dT%H:%M:%S")
    events = []
    for etype, conf in active_events.items():
        frame_b64 = _encode_frame(frame)
        events.append(_make_event(session_id, timestamp_iso, etype, conf, 0, frame_b64))

    return {
        "session_id": session_id,
        "candidate_name": candidate_name,
        "exam_name": exam_name,
        "date": capture_dt.strftime("%Y-%m-%d"),
        "duration_minutes": 0,
        "events": events,
    }


# --------------------------------------------------------------------------
# Video mode
# --------------------------------------------------------------------------
def extract_from_video(video_path, session_id, candidate_name, exam_name,
                        sample_fps=VIDEO_SAMPLE_FPS, model_name=DEFAULT_VISION_MODEL,
                        phone_conf_threshold=DEFAULT_PHONE_CONF_THRESHOLD):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_start_dt = _extract_video_start_datetime(video_path)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(round(native_fps / sample_fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_minutes = (total_frames / native_fps) / 60.0 if native_fps else 0

    estimated_calls = int(total_frames / frame_interval) if frame_interval else 0
    if estimated_calls > 60:
        logger.warning(
            f"{video_path}: sample_fps={sample_fps} implies ~{estimated_calls} vision-model calls for this "
            "video — each vision-model call can take several seconds on CPU. Consider a lower --sample-fps."
        )

    # active_events[event_type] = {"start_sec": float, "confidences": [float, ...], "frame": np.ndarray}
    active_events = {}
    finished_events = []
    frame_counter = 0

    def close_event(etype, end_sec):
        info = active_events.pop(etype)
        duration_sec = end_sec - info["start_sec"]
        if duration_sec < MIN_EVENT_DURATION_SEC.get(etype, 1.0):
            logger.debug(f"{video_path}: {etype} lasted {duration_sec:.2f}s — below debounce threshold, dropped.")
            return  # debounce: too brief to count
        avg_conf = sum(info["confidences"]) / len(info["confidences"])
        frame_b64 = _encode_frame(info["frame"])
        event_dt = video_start_dt + timedelta(seconds=info["start_sec"])
        timestamp_iso = event_dt.strftime("%Y-%m-%dT%H:%M:%S")
        finished_events.append(
            _make_event(session_id, timestamp_iso, etype, avg_conf, duration_sec * 1000, frame_b64)
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % frame_interval == 0:
            t_sec = frame_counter / native_fps

            raw_response = _call_vision_model(frame, model_name)
            vision_result = _parse_vision_response(raw_response) if raw_response is not None else _fallback_vision_result("Ollama call failed")
            current = _classify_frame(vision_result, phone_conf_threshold)

            # start/continue events
            for etype, conf in current.items():
                if etype not in active_events:
                    active_events[etype] = {"start_sec": t_sec, "confidences": [], "frame": frame.copy()}
                active_events[etype]["confidences"].append(conf)

            # close events that are no longer active
            for etype in list(active_events.keys()):
                if etype not in current:
                    close_event(etype, t_sec)

        frame_counter += 1

    # close anything still active at end of video
    end_sec = frame_counter / native_fps
    for etype in list(active_events.keys()):
        close_event(etype, end_sec)

    cap.release()

    finished_events.sort(key=lambda e: e["timestamp"])
    logger.debug(f"{video_path}: extracted {len(finished_events)} event(s) using model={model_name}")

    return {
        "session_id": session_id,
        "candidate_name": candidate_name,
        "exam_name": exam_name,
        "date": video_start_dt.strftime("%Y-%m-%d"),
        "duration_minutes": round(duration_minutes, 1),
        "events": finished_events,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Image/Video -> exam event log JSON (Ollama vision model)")
    parser.add_argument("--input", required=True, help="Path to image or video file")
    parser.add_argument("--type", required=True, choices=["image", "video"])
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--exam", required=True)
    parser.add_argument("--output", required=True, help="Where to write the session JSON")
    parser.add_argument("--sample-fps", type=float, default=VIDEO_SAMPLE_FPS, help="Video only — frames/sec sent to the vision model")
    parser.add_argument("--model", default=DEFAULT_VISION_MODEL, help="Ollama multimodal vision model to use (default: llava:7b)")
    parser.add_argument("--phone-conf-threshold", type=float, default=DEFAULT_PHONE_CONF_THRESHOLD,
                         help="Minimum model-reported confidence before a phone/device counts as an event")
    parser.add_argument("--save-to-db", action="store_true", help="Also persist the extracted events to db/proctoring.db")
    args = parser.parse_args()

    if args.type == "image":
        session_data = extract_from_image(
            args.input, args.session_id, args.candidate, args.exam,
            model_name=args.model, phone_conf_threshold=args.phone_conf_threshold,
        )
    else:
        session_data = extract_from_video(
            args.input, args.session_id, args.candidate, args.exam,
            sample_fps=args.sample_fps, model_name=args.model, phone_conf_threshold=args.phone_conf_threshold,
        )

    output_path = os.path.join(BASE_DIR, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

    if args.save_to_db:
        import db
        db.init_db()
        db.save_session_events(session_data)

    print(f"Extracted {len(session_data['events'])} event(s). Wrote session log to: {output_path}")


if __name__ == "__main__":
    main()
