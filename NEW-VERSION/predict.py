from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)
print("📷 Webcam started - Press SPACE to detect, ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    # Show live frame
    cv2.imshow("Webcam Feed", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC to quit
        print("🛑 Exiting...")
        break

    elif key == 32:  # SPACE to capture and detect
        print("📸 Capturing and processing frame...")

        # Preprocess: convert to grayscale and threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Convert single-channel thresholded image back to BGR (for YOLO)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Run YOLO detection on thresholded image
        results = model.predict(source=thresh_bgr, save=False, imgsz=640, conf=0.4)

        # Draw detections on the original thresholded frame
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = box.conf[0].item() * 100
                text = f"{label} ({conf:.1f}%)"

                # Draw bounding box and label
                cv2.rectangle(thresh_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(thresh_bgr, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show result
        cv2.imshow("Thresholded Detection", thresh_bgr)

cap.release()
cv2.destroyAllWindows()
