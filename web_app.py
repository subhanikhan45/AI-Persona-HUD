import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av, cv2, queue
from deepface import DeepFace

# 1. Page Config & State
st.set_page_config(page_title="AI Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v2.0")

if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# 2. Sidebar Controls
st.sidebar.header("System Controls")
show_hud = st.sidebar.toggle("Enable Cyber-HUD Overlay", value=True)
scanlines_active = st.sidebar.toggle("Digital Scanlines", value=True)

# 3. AI Processing Engine
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
                
                # Push data to charts
                st.session_state.result_queue.put(emotions)
                
                if show_hud:
                    # Drawing your Magenta HUD logic
                    cv2.rectangle(img, (w//2-150, h//2-150), (w//2+150, h//2+150), (255, 0, 255), 2)
                    cv2.putText(img, f"TARGET: {dominant}", (w//2-140, h//2-160), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        except:
            pass

        if scanlines_active:
            for i in range(0, h, 10):
                cv2.line(img, (0, i), (w, i), (0, 0, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Neural Feed")
    ctx = webrtc_streamer(
        key="neural-persona",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("📊 Emotion Analytics")
    chart_placeholder = st.empty()
    
    # This loop keeps the charts live
    while ctx.state.playing:
        try:
            data = st.session_state.result_queue.get(timeout=0.1)
            chart_placeholder.bar_chart(data)
        except queue.Empty:
            continue
