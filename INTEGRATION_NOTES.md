# Where these files go

Based on your two screenshots, your current layout is:

```
<parent folder>/
├── rag_system/                     <- screenshot 1's contents live here
│   ├── .venv/
│   ├── data/
│   ├── db/
│   ├── knowledge_base/
│   ├── rag_reporter.py
│   ├── report_test_output.md
│   ├── requirements.txt
│   ├── setup.bat
│   ├── test_rag.py
│   └── walkthrough.md
├── implementation_plan.md
├── offilne proctoring.png
├── Offline AI Proctoring System.pdf
├── offline-ai-proctoring-roadmap.pdf
└── rag_core.py                     <- ⚠️ this is OUTSIDE rag_system/
```

**One fix needed first:** `rag_core.py` is sitting one level above `rag_system/`.
Its imports (`chromadb`, `ollama`) and any future LangGraph code that imports
*it* only work cleanly if it's next to `rag_reporter.py`. Move it into
`rag_system/`.

**Add these 4 new files, all into `rag_system/`:**

```
rag_system/
├── extractor_agent.py          <- new
├── run_full_demo.py            <- new
├── requirements-extractor.txt  <- new
├── models/
│   ├── README.md               <- new
│   └── yolov8n.onnx            <- you add this (see models/README.md)
├── rag_core.py                 <- moved here from the parent folder
└── ...(everything else unchanged)
```

`data/frames/` will be created automatically the first time you run the
extractor — that's where it saves the snapshot image for each detected
event (matching the `frame_ref` field in your JSON schema).

# One-time setup

```powershell
cd rag_system
.venv\Scripts\activate
pip install -r requirements-extractor.txt
```

Then get `yolov8n.onnx` into `models/` — see `models/README.md`.

# Running the demo

Single command, image or video in, report out:

```powershell
python run_full_demo.py --input data/your_photo.jpg --type image ^
    --session-id exam_demo_001 --candidate "Jane Doe" --exam "CS101 Final" ^
    --report-output report_from_demo.md
```

```powershell
python run_full_demo.py --input data/your_clip.mp4 --type video ^
    --session-id exam_demo_002 --candidate "Jane Doe" --exam "CS101 Final" ^
    --report-output report_from_demo.md
```

This runs `extractor_agent.py` (image/video -> event JSON, same schema as
`data/sample_session_events.json`) then immediately hands that JSON to
`rag_reporter.py`'s existing `process_exam_session()` — no changes needed
to `rag_reporter.py` itself. Output lands in `report_from_demo.md`, same
format as `report_test_output.md`.

If you'd rather run the two steps separately (e.g. to inspect the extracted
JSON before generating the report):

```powershell
python extractor_agent.py --input data/your_photo.jpg --type image ^
    --session-id exam_demo_001 --candidate "Jane Doe" --exam "CS101 Final" ^
    --output data/session_from_image.json

python rag_reporter.py --session data/session_from_image.json --output report_from_demo.md
```

# What to know before you demo it

- **Image mode** logs each condition (no_face / multiple_faces / gaze_deviation /
  phone_detected) as a single instantaneous event with `duration_ms: 0` — a
  still photo has no time dimension, so there's nothing to debounce.
- **Video mode** samples 2 frames/sec, tracks each event type as
  start->end, and only logs it if it persists past a minimum threshold
  (1.5s gaze, 1.0s no-face/multi-face, 0.5s phone) — this is the "event
  debouncing" step the roadmap calls out, so a single blink or one bad
  frame doesn't spam the log.
- **Head pose thresholds** (yaw/pitch > 20°) are a reasonable starting
  point but not calibrated to any specific webcam/seating setup — expect to
  tune `GAZE_YAW_THRESHOLD_DEG` / `GAZE_PITCH_THRESHOLD_DEG` after your
  first real test clip.
- **Timestamps** are currently wall-clock "now" at extraction time (video
  events use the actual elapsed offset for duration, just not for the
  absolute timestamp). If you want timestamps anchored to an actual exam
  start time, pass that in and add it to `t_sec` — flag this if you want it,
  it's a small change.
- This hasn't been run against a real image/video/model on my end (no
  webcam/model file available here) — treat it as a solid first draft to
  test against your own footage, not a verified-working build.
