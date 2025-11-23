import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

st.title("✋ AirDraw AI - Real-time Sketch Recognition")

# ------------------ Load model ------------------ #
@st.cache_resource
def load_airdraw_model():
    model = load_model("airdraw_model.keras", compile=False)
    return model

model = load_airdraw_model()

# If you trained only 'star', keep just that.
# If you trained more classes, extend this list like ['cat', 'tree', 'car', 'star', 'house']
labels = ['star']


# ------------------ UI elements ------------------ #
start = st.button("Start Drawing")
clear_canvas = st.button("Clear Canvas")
frame_placeholder = st.empty()
canvas_placeholder = st.empty()
prediction_placeholder = st.empty()
image_placeholder = st.empty()

# ------------------ Session state ------------------ #
if "canvas" not in st.session_state or clear_canvas:
    st.session_state.canvas = np.zeros((256, 256), dtype=np.uint8)

canvas = st.session_state.canvas

if start:
    # Init MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
    )

    # Try multiple camera indices
    cap = None
    for idx in [0, 1, 2]:
        temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if temp_cap.isOpened():
            cap = temp_cap
            st.write(f"✅ Using camera index: {idx}")
            break

    if cap is None:
        st.error("❌ Could not open any camera. Close other apps & try again.")
    else:
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --------- MediaPipe hand tracking --------- #
            result = hands.process(rgb)
            h, w, _ = frame.shape

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Index finger tip is landmark 8
                    lm = hand_landmarks.landmark[8]

                    # Position in camera frame (for visualization)
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)

                    # Draw green dot on live camera frame
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

                    # Map to 256x256 drawing canvas coordinates
                    # lm.x and lm.y are in [0,1] relative space
                    draw_x = int(lm.x * 255)
                    draw_y = int(lm.y * 255)

                    # Draw white dots on canvas
                    cv2.circle(canvas, (draw_x, draw_y), 2, 255, -1)

            # --------- Show live camera frame --------- #
            frame_placeholder.image(rgb, channels="RGB")

            # --------- Show drawing canvas --------- #
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            canvas_placeholder.image(canvas_bgr, channels="BGR", caption="Drawing Canvas")

            # --------- Predict every few frames if canvas has content --------- #
            frame_count += 1
            if frame_count % 10 == 0:  # every 10 frames
                if np.count_nonzero(canvas) > 50:  # some drawing exists
                    # Resize to 28x28 and normalize
                    small = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
                    img_input = small.astype("float32") / 255.0
                    img_input = img_input.reshape(1, 28, 28, 1)

                    preds = model.predict(img_input)
                    idx = int(np.argmax(preds))
                    conf = float(np.max(preds))
                    label = labels[idx] if idx < len(labels) else "unknown"

                    prediction_placeholder.markdown(
                        f"### 🔮 Prediction: **{label}** (confidence: `{conf:.2f}`)"
                    )

                    # Show corresponding real image if available
                    img_path = os.path.join("real_images", f"{label}.jpg")
                    if os.path.exists(img_path):
                        image_placeholder.image(img_path, caption=f"Predicted: {label}")
                    else:
                        image_placeholder.write(f"Real image not found: {img_path}")

            # --------- Break condition for loop --------- #
            # Streamlit doesn't catch keypresses easily, so we use a limit
            if frame_count > 400:  # roughly some seconds of drawing
                st.info("⏹ Stopping after a few seconds. Click 'Start Drawing' again to continue.")
                break

        cap.release()
        hands.close()

    # Save back canvas to session_state so it's preserved across reruns
    st.session_state.canvas = canvas
