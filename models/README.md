# Getting yolov8n.onnx

`extractor_agent.py` expects a COCO-pretrained YOLOv8n model at:

    rag_system/models/yolov8n.onnx

This repo's main venv is deliberately torch-free (per walkthrough.md, to
dodge the Windows MAX_PATH bug), so **do the export in a separate, throwaway
environment** — you only need to do this once, then copy the resulting
.onnx file into this folder.

## Option A — export it yourself (one-time, separate venv)

```powershell
python -m venv export_env
export_env\Scripts\activate
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=12
:: this downloads yolov8n.pt automatically, then writes yolov8n.onnx
copy yolov8n.onnx <path-to>\rag_system\models\yolov8n.onnx
deactivate
```

You can then delete `export_env` entirely — it's not needed at runtime.
Only `onnxruntime` (already lightweight, no torch) is needed in the main
`rag_system` venv to actually *run* the model.

## Option B — download a pre-exported copy

Several public mirrors host pre-converted `yolov8n.onnx` (COCO 80-class,
640x640 input). Search "yolov8n.onnx download" and verify the source before
trusting a random binary — prefer the official Ultralytics GitHub releases
page or Hugging Face model hubs over random file-sharing links.

## Sanity check

```python
import onnxruntime as ort
sess = ort.InferenceSession("yolov8n.onnx")
print(sess.get_inputs()[0].shape)   # expect [1, 3, 640, 640]
print(sess.get_outputs()[0].shape)  # expect [1, 84, 8400]
```

`84 = 4 box coords + 80 COCO class scores`. Class id `67` is "cell phone" —
that's the only class `extractor_agent.py` filters for.
