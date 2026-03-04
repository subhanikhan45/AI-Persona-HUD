import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av, cv2, queue
from deepface import DeepFace

st.set_page_config(page_title="Neural Persona HUD", layout="wide")
st.title("🛡️ Neural Persona: Web Dashboard v5.0")

# Bridge for the Status Display
if "status_queue" not in st.session_state:
    st.session_state.status_queue = queue.Queue()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        try:
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            if results:
                face_data = results[0]['region']
                fx, fy, fw, fh = face_data['x'], face_data['y'], face_data['w'], face_data['h']
                
                dominant = results[0]['dominant_emotion'].upper()
                score = results[0]['emotion'][dominant.lower()]
                
                # Push analysis to the right-side dashboard
                st.session_state.status_queue.put({"emo": dominant, "conf": score})

                # Draw the HUD box that follows your face
                color = (255, 0, 255) # Pink
                cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, 2)
                cv2.putText(img, f"SCANNING: {dominant}", (fx, fy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except: pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Neural Feed")
    ctx = webrtc_streamer(key="neural", video_processor_factory=VideoProcessor,
                          rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))

with col2:
    st.subheader("📋 Biometric Analysis")
    
    # 1. Big Status Metric
    persona_display = st.empty()
    
    # 2. Confidence Progress Bar
    conf_display = st.empty()
    
    # 3. System Advice Card
    advice_display = st.empty()

    # The Loop that pulls AI data and puts it in the displays above
    while ctx.state.playing:
        try:
            data = st.session_state.status_queue.get(timeout=0.1)
            
            # Update the big Metric
            persona_display.metric(label="Detected Persona", value=data['emo'])
            
            # Update the progress bar
            confidence = min(float(data['conf'])/100, 1.0)
            conf_display.progress(confidence, text=f"AI Confidence: {int(data['conf'])}%")
            
            # Update the advice box based on emotion
            if data['emo'] == "HAPPY":
                advice_display.success("System Status: OPTIMAL. Positive reinforcement active.")
            elif data['emo'] == "ANGRY":
                advice_display.error("Warning: HIGH STRESS levels detected. Protocol: Breathe.")
            elif data['emo'] == "NEUTRAL":
                advice_display.info("System Status: STEADY. Monitoring baseline data.")
            else:
                advice_display.warning("Status: FLUCTUATING. Analyzing biometric spikes.")
                
        except queue.Empty: continue
