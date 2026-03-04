class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # 1. AI detects face and returns 'region' (x, y, w, h)
            results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            
            if results:
                # 2. Extract the exact face location
                face_data = results[0]['region']
                x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
                
                # 3. Get emotion for the label
                dominant = results[0]['dominant_emotion'].upper()
                
                # 4. DRAW: Use (x, y) and (w, h) so the box FOLLOWS you
                pink_color = (255, 0, 255)
                # cv2.rectangle(image, top_left, bottom_right, color, thickness)
                cv2.rectangle(img, (x, y), (x + w, y + h), pink_color, 2)
                
                # Place text just above the moving box
                cv2.putText(img, f"PERSONA: {dominant}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, pink_color, 2)
        except:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")
