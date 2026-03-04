import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av, cv2
from deepface import DeepFace

st.set_page_config(page_title="Neural Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v4.0")

# 1. Initialize the "Brain" in the dashboard memory
if "emotions" not in st.session_state:
    st.session_state["emotions"] = {"neutral": 1, "happy": 0, "angry": 0}

# 2. The Direct Camera Callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    try:
        # AI Analysis
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        if results:
            # Update the global session state directly
            st.session_state["emotions"] = results[0]['emotion']
            
            # Draw the HUD Magenta Box
            dominant = results[0]['dominant_emotion'].upper()
            cv2.rectangle(img, (170, 110), (470, 410), (255, 0, 255), 2)
            cv2.putText(img, f"PERSONA: {dominant}", (180, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    except:
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Layout: Two Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Feed")
    webrtc_streamer(
        key="direct-cam",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

with col2:
    st.subheader("📊 Live Analytics")
    # This button "forces" the page to look at the new AI data
    if st.button("Refresh Analytics"):
        st.rerun()
    
    # Draw the chart using the global memory
    st.bar_chart(st.session_state["emotions"])
