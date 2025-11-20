"""
Streamlit front-end for the lightweight object recognition demo.

Runs on the same YOLOv3-tiny + OpenCV DNN backend as the CLI script, so you only
need `opencv-python`, `numpy`, `pillow`, and `streamlit` – no CUDA/Torch.

Launch with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
from typing import Dict, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from object_recognition import COCO_LABELS, annotate, load_model, run_detection as backend_detect


st.set_page_config(page_title="Lightweight Object Recognition", layout="wide")
st.title("🔍 Lightweight Object Recognition")
st.caption("Powered by YOLOv3-tiny via OpenCV DNN — lightweight and portable.")


@st.cache_resource(show_spinner="Loading detector…")
def get_model(input_size: int):
    return load_model(input_size)


def run_detection(
    model: cv2.dnn_DetectionModel,
    frame_bgr: np.ndarray,
    threshold: float,
    nms_threshold: float,
    max_long_side: int,
):
    processed, detections = backend_detect(
        model, frame_bgr.copy(), threshold, nms_threshold, max_long_side
    )
    annotate(processed, detections, threshold)
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return rgb, detections


def format_detections(
    detections: tuple[np.ndarray, np.ndarray, np.ndarray], threshold: float
) -> List[Dict]:
    rows = []
    class_ids, scores, boxes = detections
    for label, score, (x, y, w, h) in zip(class_ids, scores, boxes):
        if score < threshold:
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        rows.append(
            {
                "label": COCO_LABELS.get(int(label), "object"),
                "score": float(score),
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1),
            }
        )
    return rows


with st.sidebar:
    st.header("Controls")
    threshold = st.slider("Score threshold", 0.05, 0.9, 0.3, 0.05)
    nms_threshold = st.slider("NMS threshold", 0.1, 0.9, 0.45, 0.05)
    input_size = st.slider("Model input (px)", 320, 640, 416, 32)
    max_long_side = st.slider("Max image size (px)", 320, 1280, 800, 40)
    model = get_model(input_size)
    st.success(f"Detector ready (YOLOv3-tiny @ {input_size}px input).")

col_upload, col_camera = st.columns(2)
with col_upload:
    uploaded = st.file_uploader("Upload an image", type=[
                                "jpg", "jpeg", "png", "bmp"])
with col_camera:
    captured = st.camera_input("…or capture from your webcam")


def file_to_bgr(file) -> np.ndarray | None:
    if file is None:
        return None
    bytes_data = file.read() if hasattr(file, "read") else file.getvalue()
    try:
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    except Exception as exc:  # pragma: no cover - Streamlit UI
        st.error(f"Unable to read image: {exc}")
        return None
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


source_bgr = file_to_bgr(uploaded) or file_to_bgr(captured)

if source_bgr is None:
    st.info("Upload an image or capture a frame to start detecting objects.")
    st.stop()

with st.spinner("Running detection…"):
    annotated_rgb, detections = run_detection(
        model, source_bgr, threshold, nms_threshold, max_long_side
    )

kept_rows = format_detections(detections, threshold)

st.subheader("Annotated image")
st.image(annotated_rgb, channels="RGB", use_column_width=True)

st.subheader(f"Detections (≥ {threshold:.2f}) — {len(kept_rows)} found")
if kept_rows:
    st.dataframe(kept_rows, use_container_width=True)
else:
    st.write(
        "No objects met the score threshold. Try lowering it or using another image.")

st.caption(
    "Tip: Lower threshold/NMS for more boxes, raise them or shrink image size for speed.")
