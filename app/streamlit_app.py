"""Streamlit demo for retail shelf empty-space detection."""
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pt"

st.set_page_config(page_title="Shelf Intelligence AI", page_icon="🛒", layout="wide")
st.title("Shelf Intelligence AI")
st.caption("Detect empty spaces on retail shelves with YOLOv8.")

@st.cache_resource(show_spinner="Loading detection model...")
def load_model(model_path: str):
    return YOLO(model_path)

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    st.caption("Lower thresholds typically detect more objects, but may increase false positives.")

uploaded = st.file_uploader("Upload a retail shelf image", type=["jpg", "jpeg", "png"])

if not MODEL_PATH.is_file():
    st.error("Model weights not found. Place your trained model at `models/best.pt`.")
    st.stop()

if uploaded is None:
    st.info("Upload an image to run detection.")
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(BytesIO(uploaded.getvalue()))).convert("RGB")
except (UnidentifiedImageError, OSError, ValueError):
    st.error("Unable to read the uploaded image. Please upload a valid JPG or PNG file.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    with st.spinner("Detecting objects..."):
        result = model.predict(
            source=np.asarray(image),
            imgsz=640,
            conf=confidence,
            iou=0.70,
            max_det=1000,
            device="cpu",
            verbose=False,
        )[0]
except Exception as exc:
    st.error(f"Detection failed: {exc}")
    st.stop()

counts = {"shelf": 0, "void": 0}
for box in result.boxes:
    name = model.names[int(box.cls.item())]
    if name in counts:
        counts[name] += 1

left, right = st.columns(2)
with left:
    st.subheader("Original image")
    st.image(image, use_container_width=True)
with right:
    st.subheader("Detected objects")
    # Ultralytics plot() returns a BGR ndarray by default.
    annotated_bgr = result.plot()
    annotated_rgb = Image.fromarray(annotated_bgr[..., ::-1])
    st.image(annotated_rgb, use_container_width=True)

metric_left, metric_right = st.columns(2)
metric_left.metric("Shelves detected", counts["shelf"])
metric_right.metric("Empty spaces detected", counts["void"])

buffer = BytesIO()
annotated_rgb.save(buffer, format="PNG")
st.download_button(
    "Download annotated image",
    data=buffer.getvalue(),
    file_name=f"{Path(uploaded.name).stem}_detections.png",
    mime="image/png",
)

st.caption("Portfolio demonstration only. Detection quality depends on image conditions and training annotations.")
