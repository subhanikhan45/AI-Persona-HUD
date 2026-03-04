import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np

# 1. UI Styling
st.set_page_config(page_title="AI Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v2.0")

# 2. Sidebar Controls
st.sidebar.header("System Controls")
show_hud = st.sidebar.toggle("Enable Cyber-HUD Overlay", value=True)
scanlines = st.sidebar.toggle("Digital Scanlines", value=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Neural Feed")
    image_placeholder = st.empty()

with col2:
    st.subheader("📊 Emotion Analytics")
    chart_placeholder = st.empty()
    status_placeholder = st.empty()

# 3. Logic
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    try:
        # Rapid AI Engine
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        emotions = results[0]['emotion']
        dominant = results[0]['dominant_emotion'].upper()

        # Update Web UI
        status_placeholder.success(f"ACTIVE PERSONA: {dominant}")
        chart_placeholder.bar_chart(emotions)

        if show_hud:
            # Draw the Magenta HUD directly on the web feed
            cv2.rectangle(frame, (w//2-150, h//2-150), (w//2+150, h//2+150), (255, 0, 255), 2)
            cv2.putText(frame, f"TARGET: {dominant}", (w//2-140, h//2-160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    except:
        pass

    # Convert to RGB for Web
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_placeholder.image(frame_rgb, channels="RGB")

cap.release()
