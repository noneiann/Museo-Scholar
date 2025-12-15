from ultralytics import YOLO
import cv2
model = YOLO("model/v3_model/runs/detect/train3/weights/best.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to exit.")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO model on the frame for inference
    results = model(frame)

    # Process results and draw bounding boxes on the frame
    for result in results:
        # Get bounding box coordinates and other info
        boxes = result.boxes
        
        # You can iterate through detections to draw custom labels/boxes
        # The 'result.plot()' method provides a simple way to visualize results
        annotated_frame = result.plot() 
        
        # Display the resulting frame
        cv2.imshow('YOLO Object Detection Live', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()