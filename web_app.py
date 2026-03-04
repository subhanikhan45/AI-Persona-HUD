import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av, cv2, queue
from deepface import DeepFace

st.set_page_config(page_title="Neural Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v6.0")

# 1. The Data Bridge (Essential for the right-side results)
if "status_queue" not in st.session_state:
    st.session_state.status_queue = queue.Queue()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # AI Logic: We use 'mediapipe' here because it is much more stable in the cloud
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe')
            
            if results and len(results) > 0:
                # Get the moving face coordinates
                face = results[0]['region']
                x, y, w, h = face['x'], face['y'], face['w'], face['h']
                
                # Get the emotion data
                dominant = results[0]['dominant_emotion'].upper()
                score = results[0]['emotion'][dominant.lower()]
                
                # Send data to the right-side dashboard
                st.session_state.status_queue.put({"emo": dominant, "conf": score})

                # DRAW: The Pink Square that FOLLOWS the face
                pink = (255, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), pink, 2)
                
                # Label box above the head
                cv2.rectangle(img, (x, y - 35), (x + w, y), pink, -1)
                cv2.putText(img, f"ID: {dominant}", (x + 5, fy - 10 if 'fy' in locals() else y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            # If AI fails, we still return the image so the cam doesn't go black
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Feed")
    ctx = webrtc_streamer(
        key="neural",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("📋 Analysis Results")
    
    # Placeholders for the right side
    res_title = st.empty()
    res_conf = st.empty()
    res_note = st.empty()

    # This loop keeps the right side alive
    while ctx.state.playing:
        try:
            # Pull data from the AI
            data = st.session_state.status_queue.get(timeout=0.1)
            
            # Show Results on the right
            res_title.metric("ACTIVE PERSONA", data['emo'])
            res_conf.progress(min(float(data['conf'])/100, 1.0), text=f"AI Confidence: {int(data['conf'])}%")
            
            # Simple status text
            if data['emo'] == "HAPPY":
                res_note.success("Status: Optimal Engagement")
            else:
                res_note.info("Status: Analyzing Biometrics...")
                
        except queue.Empty:
            continue
