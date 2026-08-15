# Extractor setup (Ollama vision model)

`extractor_agent.py` now uses a **local Ollama multimodal vision model**
(default: `llava:7b`) to inspect frames, instead of the earlier MediaPipe
Face Landmarker + YOLOv8-ONNX heuristic pipeline. This means:

- No `models/face_landmarker.task` download needed anymore.
- No `models/yolov8n.onnx` export/download needed anymore.
- `mediapipe` and `onnxruntime` are no longer required dependencies.

The only setup step is pulling the vision model through Ollama:

```powershell
ollama pull llava:7b
```

Verify it works:

```powershell
ollama run llava:7b "describe this" --image path\to\some_photo.jpg
```

If you'd rather use a different multimodal model (e.g. `llava:13b`,
`bakllava`, `moondream`), pull it and pass `--model <name>` to
`extractor_agent.py` (or `--vision-model <name>` to `run_full_demo.py`, or
set it in the Streamlit sidebar) — no code changes required.

> The previously downloaded `face_landmarker.task` / `yolov8n.onnx` files
> (and the `models/` folder) are no longer read by the pipeline and can be
> deleted if you want to reclaim the disk space.

---

## Why the switch from MediaPipe/YOLO to a vision LLM

Per the project spec (`Offline_AI_Proctoring.pdf`), Component 1 ("Image
Information Extractor Agent") is meant to use a **local Multimodal Vision
LLM** ("e.g., llava via Ollama") rather than heuristic bounding-box
models, so it can reason about context (is that a phone or a calculator?
is that gaze a stretch or something else?) rather than relying purely on
geometric thresholds. This also removes the Windows MAX_PATH / torch-free
constraints that motivated some of the earlier workarounds, since no
extra vision libraries need to be installed at all — everything runs
through the same local Ollama server already used for report generation.

Trade-off: a vision-LLM call per frame is much slower than a MediaPipe/YOLO
pass (seconds, not milliseconds, especially on CPU), so video mode defaults
to a low sample rate (`VIDEO_SAMPLE_FPS = 0.5`, i.e. one frame every 2s).
Raise `--sample-fps` for short demo clips if your hardware can keep up.
