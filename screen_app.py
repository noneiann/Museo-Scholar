from ultralytics import YOLO
import cv2
import numpy as np
from mss import mss
import time

model = YOLO("model/v4_model/runs/detect/train/weights/best.pt")

# Initialize screen capture
sct = mss()

# Get the primary monitor dimensions
monitor = sct.monitors[1]  # monitors[0] is all monitors combined, monitors[1] is primary

# Downscale factor to reduce processing load (2 = half size, 3 = third size, etc.)
DOWNSCALE_FACTOR = 2

# Process every N frames (skip frames to reduce load)
FRAME_SKIP = 2
frame_count = 0

print("Screen capture started. Press 'q' to exit.")
print(f"Capturing screen at: {monitor['width']}x{monitor['height']}")
print(f"Processing at: {monitor['width']//DOWNSCALE_FACTOR}x{monitor['height']//DOWNSCALE_FACTOR}")

annotated_frame = None

while True:
    frame_count += 1
    
    # Capture the screen
    screenshot = sct.grab(monitor)
    
    # Convert to numpy array and BGR format (OpenCV format)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Downscale to reduce processing load
    frame = cv2.resize(frame, (frame.shape[1]//DOWNSCALE_FACTOR, frame.shape[0]//DOWNSCALE_FACTOR))

    # Only process every N frames to reduce load
    if frame_count % FRAME_SKIP == 0:
        # Run YOLO model on the frame for inference
        results = model(frame, verbose=False)  # Lower image size for faster inference

        # Get annotated frame with detections (results[0] is the first result)
        annotated_frame = results[0].plot()
    
    # Display the resulting frame (only if we have one)
    if annotated_frame is not None:
        cv2.imshow('YOLO Object Detection - Screen Capture', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Delay to limit FPS (~5 FPS)
    time.sleep(0.1)

# When everything is done, destroy windows
cv2.destroyAllWindows()
