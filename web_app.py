import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
from deepface import DeepFace
import queue

# 1. Page Config
st.set_page_config(page_title="Neural Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v3.0")

# STUN server to help the video pass through cloud firewalls
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 2. THE BRIDGE: A shared queue to pass data between threads
# (The camera thread 'puts' data, the main thread 'gets' it)
result_queue = queue.Queue()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        try:
            # AI Analysis
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            if results:
                emotions = results[0]['emotion']
                dominant = results[0]['dominant_emotion'].upper()
                
                # Push analysis results to the main thread
                result_queue.put(emotions)

                # Draw the HUD
                cv2.rectangle(img, (w//2-130, h//2-130), (w//2+130, h//2+130), (255,0,255), 2)
                cv2.putText(img, f"PERSONA: {dominant}", (w//2-120, h//2-140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
        except:
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Start the video stream
    ctx = webrtc_streamer(
        key="neural-persona",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 Live Analytics")
    # This acts as a container that we will overwrite in the loop below
    chart_placeholder = st.empty()

# 4. THE LIVE UPDATE LOOP
# This part runs on the Main Thread and watches the queue for new AI data
while ctx.state.playing:
    try:
        # Check for new emotion data (wait up to 0.1 seconds)
        latest_emotions = result_queue.get(timeout=0.1)
        if latest_emotions:
            # Update the chart container with new data
            chart_placeholder.bar_chart(latest_emotions)
    except queue.Empty:
        # If no new data yet, just keep waiting
        continue
