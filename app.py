import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Bangla Handwritten Word Recognition", page_icon="✍️", layout="centered")

@st.cache_resource
def load_model():
    path = Path("models/model.keras")
    if not path.exists():
        return None
    return tf.keras.models.load_model(path)

def normalize_canvas(rgba):
    rgb = np.asarray(rgba, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGBA2GRAY)
    # Canvas background is white; ink is black.
    _, ink = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return ink

def crop_pad_resize(binary, size=64):
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    crop = binary[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape
    scale = 48.0 / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x0, y0 = (size - nw)//2, (size - nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas.astype("float32")[..., None] / 255.0

def split_word(ink, expected_units=None):
    # Projection-based heuristic. Bangla matra can connect letters; this is not
    # a linguistically complete segmentation algorithm.
    ys, xs = np.where(ink > 0)
    if len(xs) == 0:
        return []
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    word = ink[y0:y1, x0:x1]
    projection = (word > 0).sum(axis=0).astype(np.float32)
    smooth = np.convolve(projection, np.ones(5)/5, mode="same")
    threshold = max(1.0, float(smooth.max()) * 0.035)
    active = smooth > threshold
    spans, start = [], None
    for i, val in enumerate(active):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= 2:
                spans.append([start, i])
            start = None
    if start is not None and len(active)-start >= 2:
        spans.append([start, len(active)])
    # Merge tiny gaps; keep a minimum width to avoid over-segmentation.
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= 2:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if expected_units and len(merged) > expected_units:
        # Merge adjacent segments across the smallest projection valleys.
        while len(merged) > expected_units:
            gaps = [merged[i+1][0] - merged[i][1] for i in range(len(merged)-1)]
            idx = int(np.argmin(gaps))
            merged[idx][1] = merged[idx+1][1]
            merged.pop(idx+1)
    parts = []
    for a, b in merged:
        piece = word[:, max(0, a-2):min(word.shape[1], b+2)]
        # Attach matra/headline strokes crossing above the segment using a
        # generous full-height crop, preserving the whole vertical extent.
        if np.count_nonzero(piece) > 5:
            parts.append(piece)
    return parts

def predict_word(model, parts, labels):
    output, details = [], []
    for part in parts:
        sample = crop_pad_resize(part)
        if sample is None:
            continue
        probs = model.predict(sample[None, ...], verbose=0)[0]
        idx = int(np.argmax(probs))
        output.append(labels.get(str(idx), "�"))
        details.append((labels.get(str(idx), f"class_{idx}"), float(probs[idx])))
    return "".join(output), details

st.title("বাংলা হাতের লেখা শনাক্তকরণ")
st.caption("Draw one Bangla word below. The system returns a combined word-level string.")
model = load_model()
if model is None:
    st.warning("No trained model found. Download the dataset and run `python train.py` first.")
    st.stop()

with open("labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

st.markdown("**Write one word from left to right.** For best results, leave small gaps between character units.")
expected = st.number_input("Approximate number of character/jukto units (optional)", min_value=0, max_value=12, value=0, help="Set 0 to let the segmentation heuristic decide.")
canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=8,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=220,
    width=650,
    drawing_mode="freedraw",
    key="bangla_word_canvas",
    display_toolbar=True,
)
if st.button("Recognize whole word", type="primary", use_container_width=True):
    if canvas.image_data is None or not np.any(np.asarray(canvas.image_data)[:, :, 3] > 0):
        st.info("Please draw a word first.")
    else:
        rgba = np.asarray(canvas.image_data, dtype=np.uint8)
        ink = normalize_canvas(rgba)
        parts = split_word(ink, expected if expected > 0 else None)
        if not parts:
            st.warning("No handwriting detected.")
        else:
            word, details = predict_word(model, parts, labels)
            st.subheader("Predicted word")
            st.markdown(f"<div style='font-size:42px;font-weight:700;line-height:1.5'>{word}</div>", unsafe_allow_html=True)
            st.caption("Combined output from left-to-right unit predictions; this is not an end-to-end word model.")
            with st.expander("Unit predictions and confidence"):
                for i, (char, confidence) in enumerate(details, 1):
                    st.write(f"{i}. {char} — {confidence:.1%}")
            st.image(Image.fromarray(ink), caption="Preprocessed ink mask", clamp=True)
st.divider()
st.caption("Research prototype: isolated-character training data does not provide word-level labels. Connected Bangla handwriting, vowel signs (কার), and ligatures may be segmented or recognized incorrectly.")
