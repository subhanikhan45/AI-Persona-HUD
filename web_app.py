import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av, cv2, queue
from deepface import DeepFace

# 1. Page Configuration
st.set_page_config(page_title="AI Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v4.0")

# Shared Queue for Thread Communication (Bridge)
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# 2. Sidebar System Controls
st.sidebar.header("🕹️ System Controls")
show_hud = st.sidebar.toggle("Enable Dynamic Face Tracking", value=True)
scanlines_active = st.sidebar.toggle("Digital Scanlines", value=True)
st.sidebar.markdown("---")
st.sidebar.write("Developer: **Ashraf**")
st.sidebar.write("Hardware: **HP EliteBook 840 G5**")

# 3. AI Processing Engine with Dynamic Tracking
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        try:
            # AI Analysis (Using OpenCV backend for speed)
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            if results:
                # Get the Dynamic Face Region (This is what makes the box follow you)
                face_data = results[0]['region']
                fx, fy, fw, fh = face_data['x'], face_data['y'], face_data['w'], face_data['h']
                
                emotions = results[0]['emotion']
                dominant = results[0]['dominant_emotion'].upper()
                
                # Push data to charts in the dashboard
                st.session_state.result_queue.put(emotions)
                
                if show_hud:
                    # DRAW: The box now uses coordinates from 'face_data'
                    color = (255, 0, 255) # Pink/Magenta
                    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, 2)
                    
                    # Label follows the top of the box
                    cv2.rectangle(img, (fx, fy - 35), (fx + fw, fy), color, -1)
                    cv2.putText(img, f"PERSONA: {dominant}", (fx + 5, fy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except:
            pass

        # Apply Visual Scanlines if toggled
        if scanlines_active:
            for i in range(0, h, 10):
                cv2.line(img, (0, i), (w, i), (0, 0, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Professional Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Neural Feed")
    # STUN servers allow video to pass through corporate/school firewalls
    ctx = webrtc_streamer(
        key="neural-hud",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("📊 Emotion Analytics")
    chart_placeholder = st.empty()
    
    # Sync Loop: Updates charts ONLY when the camera is running
    while ctx.state.playing:
        try:
            # Pull the data from the AI thread
            data = st.session_state.result_queue.get(timeout=0.1)
            chart_placeholder.bar_chart(data)
        except queue.Empty:
            continue
