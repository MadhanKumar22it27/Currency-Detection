from ultralytics import YOLO
import cv2
import pyttsx3

# ✅ Load your trained model (change to best.pt path if needed)
model = YOLO("best.pt")  # After training

# ✅ Initialize text-to-speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ✅ Load image
image_path = "E:/technical/Innovation/Currency-project/Currency-Detector/Hundred_rupee.jpg"
results = model(image_path)

# ✅ Draw results and speak detected notes
img = cv2.imread(image_path)
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        confidence = box.conf[0].item() * 100
        text = f"{label} detected ({confidence:.1f}%)"
        speak(text)
        print("💰", text)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# from ultralytics import YOLO
# import cv2
# import pyttsx3

# # ✅ Load your trained YOLO model
# model = YOLO("best.pt")  # Replace with your custom model path if different

# # ✅ Initialize text-to-speech
# engine = pyttsx3.init()

# def speak(text):
#     engine.say(text)
#     engine.runAndWait()

# # ✅ Start webcam
# cap = cv2.VideoCapture(0)
# print("📸 Webcam started - Press SPACE to detect, ESC to exit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Failed to grab frame")
#         break

#     # Show live webcam feed
#     cv2.imshow("Live Feed - Press SPACE to detect", frame)

#     key = cv2.waitKey(1) & 0xFF

#     if key == 32:  # SPACE key pressed
#         results = model(frame)[0]
#         for box in results.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cls = int(box.cls[0])
#             label = model.names[cls]
#             confidence = box.conf[0].item() * 100
#             text = f"{label} detected ({confidence:.1f}%)"
#             print("💰", text)
#             speak(text)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, text, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         # Show the prediction
#         cv2.imshow("Prediction", frame)
#         cv2.waitKey(2000)  # Display result for 2 seconds

#     elif key == 27:  # ESC key to exit
#         print("🔴 Exiting...")
#         break

# cap.release()
# cv2.destroyAllWindows()
