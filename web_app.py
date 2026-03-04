import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av, cv2, queue
from deepface import DeepFace

# 1. THE BRIDGE: Create a queue to hold the emotion results
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # AI Analysis
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            if results:
                face_data = results[0]['region']
                x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
                
                # Get the results
                emotions = results[0]['emotion'] # The full breakdown (0-100%)
                dominant = results[0]['dominant_emotion'].upper()
                
                # PUSH the results into the bridge for the dashboard to see
                st.session_state.result_queue.put({"emo": dominant, "all": emotions})
                
                # DRAW: Pink Tracking Box
                pink_color = (255, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), pink_color, 2)
                cv2.putText(img, f"PERSONA: {dominant}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, pink_color, 2)
        except:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 2. DASHBOARD LAYOUT
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Feed")
    ctx = webrtc_streamer(
        key="neural-hud",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("📊 Emotion Analytics")
    # Placeholders for results
    persona_text = st.empty()
    confidence_chart = st.empty()
    status_msg = st.empty()

    # 3. THE SYNC LOOP: Pulls data from the bridge and updates the UI
    while ctx.state.playing:
        try:
            # Pull data from the queue (wait up to 0.1 seconds)
            data = st.session_state.result_queue.get(timeout=0.1)
            
            # SHOW the results properly
            persona_text.metric("Active Persona", data['emo'])
            confidence_chart.bar_chart(data['all'])
            
            # Add a system status note
            if data['emo'] == "HAPPY":
                status_msg.success("System Status: OPTIMAL")
            else:
                status_msg.info("System Status: SCANNING...")
                
        except queue.Empty:
            continue
