import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
from deepface import DeepFace
import queue

# 1. Page Configuration
st.set_page_config(page_title="Neural Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v3.0")

# ICE Servers allow the video to bypass firewalls in the cloud
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Shared Queue to pass data from AI thread to UI thread
result_queue = queue.Queue()

# 2. The AI Video Processor
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        try:
            # Rapid AI Engine
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            if results:
                emotions = results[0]['emotion']
                dominant = results[0]['dominant_emotion'].upper()
                
                # PUSH the data into the queue for the chart
                result_queue.put(emotions)

                # Draw the HUD overlay (Magenta)
                color = (255, 0, 255)
                cv2.rectangle(img, (w//2-130, h//2-130), (w//2+130, h//2+130), color, 2)
                cv2.putText(img, f"TARGET: {dominant}", (w//2-120, h//2-140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Biometric Feed")
    webrtc_ctx = webrtc_streamer(
        key="neural-hud",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 Live Emotion Analytics")
    chart_placeholder = st.empty()
    
    # 4. THE UI UPDATE LOOP
    # This keeps the main thread alive and pulls data from the queue
    while webrtc_ctx.state.playing:
        try:
            # Pull latest data from the queue
            data = result_queue.get(timeout=1.0)
            chart_placeholder.bar_chart(data)
        except queue.Empty:
            continue
