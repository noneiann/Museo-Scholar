from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import base64
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize YOLO model
model = YOLO('model/v4_model/runs/detect/train/weights/best.pt')

# Initialize Gemini (set your API key in environment variable)
genai.configure(api_key=os.getenv('API_KEY'))
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
print(genai.list_models())
# Global variables to store current frame and detections
current_frame = None
current_detections = []

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # Set webcam to 15 FPS for lighter performance
        self.video.set(cv2.CAP_PROP_FPS, 15)
        self.frame_count = 0
        self.detection_interval = 2  # Run detection every 3 frames for lighter performance
        self.last_boxes = []
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        global current_frame, current_detections
        
        ret, frame = self.video.read()
        if not ret:
            return None
        
        self.frame_count += 1
        
        # Run YOLO detection only every N frames
        if self.frame_count % self.detection_interval == 0:
            # Run YOLO detection with smaller image size for better performance
            results = model(frame, verbose=False, imgsz=640)
            
            # Store detections for click handling
            current_frame = frame.copy()
            current_detections = []
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    
                    # Store detection info
                    current_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': class_name
                    })
                
                # Get annotated frame
                annotated_frame = result.plot()
                self.last_boxes = annotated_frame
        else:
            # Use cached detections for smooth video
            if hasattr(self, 'last_boxes') and len(self.last_boxes) > 0:
                # Draw cached boxes on current frame
                results = model(frame, verbose=False, imgsz=416)
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
        
        # Encode frame as JPEG with moderate quality for better performance
        ret, jpeg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes()

def generate_frames():
    camera = VideoCamera()
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('webcam_ui.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_object_info', methods=['POST'])
def get_object_info():
    global current_frame, current_detections
    
    data = request.json
    click_x = data['x']
    click_y = data['y']
    
    # Find which bounding box was clicked
    clicked_object = None
    for detection in current_detections:
        x1, y1, x2, y2 = detection['bbox']
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            clicked_object = detection
            break
    
    if clicked_object is None:
        return jsonify({'error': 'No object found at clicked location'})
    
    # Crop the object from the frame
    x1, y1, x2, y2 = clicked_object['bbox']
    cropped_object = current_frame[y1:y2, x1:x2]
    
    # Convert cropped object to PIL Image for Gemini
    cropped_rgb = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cropped_rgb)
    
    # Get multimodal details from Gemini
    try:
        prompt = f"""This is a detected object classified as '{clicked_object['class']}'. 
        Please provide detailed information about this object, including:
        1) What you see in the image
        2) Historical/cultural significance if applicable (especially for museum artifacts like the Golden Tara or Manunggul Jar)
        3) Physical characteristics
        4) Any interesting facts or context
        
        Keep the response concise but informative (around 200-300 words)."""
        
        response = gemini_model.generate_content([prompt, pil_image])
        ai_description = response.text
        
        return jsonify({
            'class': clicked_object['class'],
            'confidence': clicked_object['confidence'],
            'description': ai_description,
            'bbox': clicked_object['bbox']
        })
    
    except Exception as e:
        return jsonify({
            'class': clicked_object['class'],
            'confidence': clicked_object['confidence'],
            'description': f"AI description unavailable. Error: {str(e)}",
            'bbox': clicked_object['bbox']
        })

if __name__ == '__main__':
    print("Starting web server...")
    print("Open http://localhost:5000 in your browser")
    print("Make sure to set GEMINI_API_KEY environment variable for AI descriptions")
    app.run(debug=True, threaded=True)
