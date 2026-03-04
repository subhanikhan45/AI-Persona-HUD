import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
from deepface import DeepFace
import numpy as np

# 1. Page Config & Professional UI
st.set_page_config(page_title="Neural Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v2.0")
st.markdown("---")

# ICE Servers allow the video to bypass firewalls (Essential for Cloud)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 2. Sidebar Controls
st.sidebar.header("System Controls")
show_hud = st.sidebar.toggle("Enable Cyber-HUD Overlay", value=True)
persona_mode = st.sidebar.selectbox("HUD Mode", ["Standard", "Targeting", "Biometric"])

# 3. The AI Video Processor Class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_persona = "SCANNING..."
        self.current_color = (255, 0, 255) # Default Magenta

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        try:
            # Rapid AI Engine (Optimized for WebRTC)
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            if results:
                emotion = results[0]['dominant_emotion'].upper()
                
                # Persona Mapping
                if emotion == 'ANGRY':
                    self.current_persona = "THE WARRIOR"
                    self.current_color = (0, 0, 255) # Red
                elif emotion == 'HAPPY':
                    self.current_persona = "THE OPTIMIST"
                    self.current_color = (255, 255, 0) # Cyan
                else:
                    self.current_persona = "THE ARCHITECT"
                    self.current_color = (255, 191, 0) # Sky Blue

                if show_hud:
                    # Draw HUD Bounding Box
                    gap = 130
                    cx, cy = w // 2, h // 2
                    cv2.rectangle(img, (cx-gap, cy-gap), (cx+gap, cy+gap), self.current_color, 2)
                    
                    # Draw Persona Label
                    cv2.rectangle(img, (cx-gap, cy-gap-40), (cx+gap, cy-gap), (0,0,0), -1)
                    cv2.putText(img, self.current_persona, (cx-gap+10, cy-gap-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except:
            pass

        # Cyber-Scanlines Overlay
        for i in range(0, h, 10):
            cv2.line(img, (0, i), (w, i), (0, 0, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Dashboard Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📡 Live Biometric Stream")
    webrtc_streamer(
        key="persona-hud",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📋 System Status")
    st.info("System: Online")
    st.write("Hardware: HP EliteBook 840 G5 (Client)")
    st.write("AI Engine: DeepFace v0.0.92")
    st.warning("Note: First start may take 1-2 minutes to load AI models.")
