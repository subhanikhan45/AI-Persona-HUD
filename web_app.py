import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av, cv2, queue
from deepface import DeepFace

st.set_page_config(page_title="Neural Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v3.0")

# 1. The Data Bridge (Queue)
# This allows the AI thread to send data to the Chart thread
result_queue = queue.Queue()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        try:
            # AI Analysis
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            if results:
                emotions = results[0]['emotion']
                # PUSH data to the bridge
                result_queue.put(emotions)
                
                # Draw the HUD box you already have
                dominant = results[0]['dominant_emotion'].upper()
                cv2.rectangle(img, (170, 110), (470, 410), (255, 0, 255), 2)
                cv2.putText(img, f"PERSONA: {dominant}", (180, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        except: 
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 2. Layout (Camera on Left, Chart on Right)
col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(
        key="neural",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("📊 Live Analytics")
    # This empty container will be updated by the loop below
    chart_placeholder = st.empty()

# 3. THE LIVE UPDATE LOOP
# This part stays "awake" to catch data from the AI and update the chart
while ctx.state.playing:
    try:
        # Get the latest emotion data from the bridge
        data = result_queue.get(timeout=0.1)
        # Update ONLY the chart on the right
        chart_placeholder.bar_chart(data)
    except queue.Empty:
        continue
