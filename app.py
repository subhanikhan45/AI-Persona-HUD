import cv2
from deepface import DeepFace
import collections
import numpy as np
import time
import winsound

# 1. SETUP: Persona Names & Color Schemes (BGR Format)
PERSONA_CONFIG = {
    'happy':    {'name': 'THE OPTIMIST',    'color': (255, 255, 0)},   # Cyan-ish
    'neutral':  {'name': 'THE ARCHITECT',   'color': (255, 191, 0)},  # Deep Sky Blue
    'angry':    {'name': 'THE WARRIOR',     'color': (0, 0, 255)},    # Red
    'sad':      {'name': 'THE PHILOSOPHER', 'color': (200, 0, 100)},  # Purple
    'surprise': {'name': 'THE EXPLORER',    'color': (0, 255, 255)},  # Yellow
}

# Initialize Camera
cap = cv2.VideoCapture(0)
emotion_buffer = collections.deque(maxlen=15) # For smooth transitions

print(">>> HUD BOOTING UP... SCANNING FOR BIOMETRICS")

prev_persona = ""

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # Mirror Mode
    h, w, _ = frame.shape
    current_time = time.time()

    try:
        # RAPID AI ENGINE (Optimized for your i5 CPU)
        # Using 'opencv' backend is the secret to keeping it "Rapid"
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        
        if results and len(results) > 0:
            all_emotions = results[0]['emotion']
            main_emo = results[0]['dominant_emotion']
            emotion_buffer.append(main_emo)
            
            # Get stable emotion to avoid flickering text
            stable_emo = max(set(emotion_buffer), key=list(emotion_buffer).count)
            config = PERSONA_CONFIG.get(stable_emo, {'name': 'SCANNING...', 'color': (255, 0, 255)})
            
            current_color = config['color']
            persona_name = config['name']

            # 1. PERSONA SWITCH SOUND (winsound)
            if persona_name != prev_persona:
                winsound.Beep(1000, 100)
                prev_persona = persona_name

            # --- DRAWING THE HUD ELEMENTS ---

            # 2. BLINKING "REC" INDICATOR (Top Right)
            if int(current_time * 2) % 2 == 0: # Faster blink for high-tech feel
                cv2.circle(frame, (w - 30, 30), 8, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (w - 85, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 3. DYNAMIC BOUNDING BOX (Corner HUD Style)
            cx, cy = w // 2, h // 2
            gap = 130
            # Draw 4 corners
            pts = [
                [(cx-gap, cy-gap), (cx-gap+30, cy-gap), (cx-gap, cy-gap+30)], # TL
                [(cx+gap, cy-gap), (cx+gap-30, cy-gap), (cx+gap, cy-gap+30)], # TR
                [(cx-gap, cy+gap), (cx-gap+30, cy+gap), (cx-gap, cy+gap-30)], # BL
                [(cx+gap, cy+gap), (cx+gap-30, cy+gap), (cx+gap, cy+gap-30)]  # BR
            ]
            for p in pts:
                cv2.line(frame, p[0], p[1], current_color, 2)
                cv2.line(frame, p[0], p[2], current_color, 2)

            # 4. THE PERSONA BANNER (Top Center)
            cv2.rectangle(frame, (w//2-160, 20), (w//2+160, 65), (0,0,0), -1)
            cv2.rectangle(frame, (w//2-160, 20), (w//2+160, 65), current_color, 1)
            cv2.putText(frame, persona_name, (w//2-140, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # 5. LIVE VIBE BARS (Sidebar Data)
            for i, (em, score) in enumerate(all_emotions.items()):
                bar_w = int(score * 1.5)
                y_pos = 120 + (i * 35)
                cv2.putText(frame, em.upper(), (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.rectangle(frame, (20, y_pos+5), (170, y_pos+15), (40,40,40), -1) # Track
                cv2.rectangle(frame, (20, y_pos+5), (20+bar_w, y_pos+15), current_color, -1) # Fill

    except Exception as e:
        # If no face is found, display scanning message
        cv2.putText(frame, "BIOMETRIC SEARCHING...", (w//2-120, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 6. DIGITAL SCANLINES (The "Cyberpunk" Overlay)
    for i in range(0, h, 10):
        cv2.line(frame, (0, i), (w, i), (0, 0, 0), 1)

    # Show Final Window
    cv2.imshow('NEURAL PERSONA HUD v1.0', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()