import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av, cv2, queue
from deepface import DeepFace

st.set_page_config(page_title="Neural Persona HUD", layout="wide")
st.title("🛡️Emotion Live Detector")

# 1. The Queue Bridge
result_queue = queue.Queue()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        try:
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            if results:
                emotions = results[0]['emotion']
                # PUSH data to the queue
                result_queue.put(emotions)
                
                # Draw HUD
                dominant = results[0]['dominant_emotion'].upper()
                cv2.rectangle(img, (170, 110), (470, 410), (255, 0, 255), 2)
                cv2.putText(img, f"PERSONA: {dominant}", (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        except: pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 2. Layout (The Fix is Here)
col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(key="neural", video_processor_factory=VideoProcessor, 
                          rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                          media_stream_constraints={"video": True, "audio": False})

with col2:
    st.subheader("📊 Live Analytics")
    chart_placeholder = st.empty()
    
    # This simplified loop ensures the chart updates ONLY when data exists
    while ctx.state.playing:
        try:
            # We use a very short timeout to keep the UI responsive
            data = result_queue.get(timeout=0.1)
            chart_placeholder.bar_chart(data)
        except queue.Empty:
            continue



