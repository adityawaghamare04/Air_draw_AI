import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import os

st.set_page_config(page_title="AirDraw AI", page_icon="✋", layout="centered")
st.title("✋ AirDraw AI – Sketch Recognition (Cloud Version)")

st.write(
    """
    Draw an object in the canvas below (e.g., **apple**, **house**, **tree**) and 
    the model will try to recognize it.
    """
)

# --------- Load model once --------- #
@st.cache_resource
def load_airdraw_model():
    model = load_model("airdraw_model.keras", compile=False)
    return model

model = load_airdraw_model()

# Classes your model was trained on
labels = ["apple", "house", "tree"]

# --------- Canvas settings --------- #
canvas_size = 256

st.subheader("✏️ Draw here")
st.caption("Use white pen on black background. Click 'Clear' to reset.")

# Canvas component
canvas_result = st_canvas(
    fill_color="#000000",        # background fill
    stroke_width=10,             # pen thickness
    stroke_color="#FFFFFF",      # white pen
    background_color="#000000",  # black background
    width=canvas_size,
    height=canvas_size,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)
with col1:
    clear = st.button("🧹 Clear")
with col2:
    predict_btn = st.button("🔮 Predict")

# Handle clear: just rerun, canvas resets automatically
if clear:
    st.experimental_rerun()

prediction_placeholder = st.empty()
image_placeholder = st.empty()

# --------- Prediction logic --------- #
if predict_btn:
    if canvas_result.image_data is None:
        st.warning("Draw something first on the canvas!")
    else:
        # Get RGBA image from canvas
        img_data = canvas_result.image_data.astype("uint8")
        img = Image.fromarray(img_data)

        # Convert to grayscale
        img = img.convert("L")  # L = (8-bit pixels, black and white)

        # Invert if needed: we drew white on black; this makes it black on white
        img = ImageOps.invert(img)

        # Resize to 28x28 as expected by the CNN
        img_small = img.resize((28, 28), resample=Image.BILINEAR)

        # Convert to numpy and normalize
        arr = np.array(img_small).astype("float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        # Run prediction
        preds = model.predict(arr)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        label = labels[idx] if idx < len(labels) else "unknown"

        prediction_placeholder.markdown(
            f"### 🔮 Prediction: **{label}**  \nConfidence: `{conf:.2f}`"
        )

        # Show 28x28 processed image as a preview
        st.subheader("📏 Model Input (28×28)")
        st.image(img_small.resize((140, 140), resample=Image.NEAREST), caption="What the model sees", use_column_width=False)

        # Show corresponding real-world image if available
        img_path = os.path.join("real_images", f"{label}.jpg")
        if os.path.exists(img_path):
            st.subheader("📷 Real Image")
            image_placeholder.image(img_path, caption=f"Predicted: {label}")
        else:
            st.info(f"No real image found at `{img_path}`. Add one if you want to display it.")
