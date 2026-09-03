import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "generator_sketch2image.h5"
IMAGE_SIZE = 128

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------------------
# CREATE SKETCH (same as training)
# ---------------------------
def create_sketch(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    sketch = edges.astype(np.float32) / 255.0
    sketch = np.expand_dims(sketch, axis=-1)
    return sketch

# ---------------------------
# PREPROCESS
# ---------------------------
def preprocess_image(image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np = np.array(image)

    sketch = create_sketch(image_np)

    # Normalize to [-1, 1] (IMPORTANT)
    sketch = (sketch * 2.0) - 1.0

    sketch = np.expand_dims(sketch, axis=0)
    return sketch, image_np

# ---------------------------
# POSTPROCESS OUTPUT
# ---------------------------
def postprocess_image(output):
    output = output[0]  # remove batch

    # Convert from [-1,1] → [0,255]
    output = (output + 1.0) / 2.0
    output = np.clip(output, 0, 1)

    output = (output * 255).astype(np.uint8)
    return output

# ---------------------------
# UI
# ---------------------------
st.title("🧠 Sketch → Real Image Generator")
st.write("Upload a human image → model converts it into a realistic version via sketch translation.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    sketch_input, original = preprocess_image(image)

    # Show original
    st.subheader("Original Image")
    st.image(original, use_column_width=True)

    # Show sketch
    sketch_display = (sketch_input[0] + 1.0) / 2.0
    st.subheader("Generated Sketch (Input to Model)")
    st.image(sketch_display, use_column_width=True, clamp=True)

    # Predict
    if st.button("Generate Image"):
        with st.spinner("Generating..."):
            output = model.predict(sketch_input)

            generated_image = postprocess_image(output)

            st.subheader("Generated Image")
            st.image(generated_image, use_column_width=True)