# cam_test_streamlit.py
import streamlit as st
import cv2

st.title("Camera Test")

start = st.button("Start Camera")
frame_placeholder = st.empty()

if start:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Cannot open camera at index 0")
    else:
        for i in range(200):
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")
        cap.release()
