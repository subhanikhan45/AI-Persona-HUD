import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av, cv2, queue
from deepface import DeepFace

st.set_page_config(page_title="Neural Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v7.0")

# 1. The Data Bridge
if "status_queue" not in st.session_state:
    st.session_state.status_queue = queue.Queue()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # We use enforce_detection=False to prevent the 'No face detected' crash
            # We use detector_backend='opencv' because it is the LIGHTEST for your 8GB RAM
            results = DeepFace.analyze(img, actions=['emotion'], 
                                       enforce_detection=False, 
                                       detector_backend='opencv',
                                       align=False) # Align=False makes it 2x faster
            
            if results and len(results) > 0:
                face = results[0]['region']
                # The exact coordinates of your face
                x, y, w, h = int(face['x']), int(face['y']), int(face['w']), int(face['h'])
                
                dominant = results[0]['dominant_emotion'].upper()
                score = results[0]['emotion'][dominant.lower()]
                
                # Update the right-side dashboard
                st.session_state.status_queue.put({"emo": dominant, "conf": score})

                # DRAWING THE BOX:
                pink = (255, 0, 255)
                # Draw the main tracking square
                cv2.rectangle(img, (x, y), (x + w, y + h), pink, 2)
                # Draw a small header box for the text
                cv2.rectangle(img, (x, y - 30), (x + w, y), pink, -1)
                cv2.putText(img, dominant, (x + 5, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            # If the AI fails, we still return the frame so you see the camera
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
    st.subheader("📋 Live Results")
    res_title = st.empty()
    res_conf = st.empty()

    while ctx.state.playing:
        try:
            data = st.session_state.status_queue.get(timeout=0.1)
            res_title.metric("IDENTIFIED PERSONA", data['emo'])
            res_conf.progress(min(float(data['conf'])/100, 1.0), text=f"Confidence: {int(data['conf'])}%")
        except queue.Empty:
            continue
