############################################Version - 1#########################################################

# import tensorflow as tf
# import numpy as np
# import cv2
# import os

# # Load the trained model
# model = tf.keras.models.load_model("currency_model-2.h5")
# print("✅ Model Loaded Successfully!")

# # Define image size
# IMG_SIZE = (224, 224)

# # Define class labels (ensure they match training class names)
# class_labels = ['10', '20', '50', '100', '200', '500', '2000']  # Update as per your dataset folder names

# # Function to preprocess the image
# def preprocess_image(image_path):
#     img = cv2.imread(image_path)  # Read image
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     img = cv2.resize(img, IMG_SIZE)  # Resize to match model input
#     img = img / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
#     return img

# # Function to predict currency value
# def predict_currency(image_path):
#     img = preprocess_image(image_path)  # Preprocess image
#     prediction = model.predict(img)  # Get model prediction
#     predicted_class = np.argmax(prediction)  # Get the index of highest probability
#     currency_value = class_labels[predicted_class]  # Get the corresponding currency value
#     confidence = np.max(prediction) * 100  # Confidence percentage

#     print(f"💰 Predicted Currency: {currency_value} INR")
#     print(f"🔹 Confidence: {confidence:.2f}%")

# image_path = "Ten_rupee.jpg"  
# predict_currency(image_path)

#####################################Version - 2###############################################################
# import tensorflow as tf
# import numpy as np
# import cv2

# # Load the trained model
# model = tf.keras.models.load_model("Currency_model.h5")
# print("✅ Model Loaded Successfully!")

# # Define image size
# IMG_SIZE = (224, 224)

# # Define class labels (ensure they match training class names)
# class_labels = ['10', '20', '50', '100', '200', '500', '2000']  # Update based on your dataset

# # Function to preprocess the image
# def preprocess_image(image):
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     img = cv2.resize(img, IMG_SIZE)  # Resize to match model input
#     img = img / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
#     return img

# # Function to predict currency value
# def predict_currency(image):
#     img = preprocess_image(image)  # Preprocess image
#     prediction = model.predict(img)  # Get model prediction
#     predicted_class = np.argmax(prediction)  # Get the index of highest probability
#     currency_value = class_labels[predicted_class]  # Get the corresponding currency value
#     confidence = np.max(prediction) * 100  # Confidence percentage
#     return currency_value, confidence

# # Open webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("❌ Error: Could not open webcam.")
#     exit()

# print("📷 Press SPACE to capture the image.")
# print("❌ Press ESC to exit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Error: Failed to capture image.")
#         break

#     # Show live video feed
#     cv2.imshow("Press SPACE to Capture", frame)

#     key = cv2.waitKey(1) & 0xFF

#     if key == 27:  # ESC key to exit
#         break
#     elif key == 32:  # SPACE key to capture and predict
#         captured_image = frame.copy()  # Copy frame for processing
#         currency, confidence = predict_currency(captured_image)

#         # Overlay prediction on image
#         text = f"Currency: {currency} INR ({confidence:.2f}%)"
#         cv2.putText(captured_image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Show the predicted image
#         cv2.imshow("Prediction", captured_image)
#         print(f"💰 Predicted Currency: {currency} INR")
#         print(f"🔹 Confidence: {confidence:.2f}%")

#         # Wait for key press to continue
#         cv2.waitKey(0)

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

#############################  Version - 3  ###################################################################

# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# from flask import Flask, request, jsonify

# # ✅ Load the trained model
# model = tf.keras.models.load_model("currency_model.keras")
# print("✅ Model Loaded Successfully!")

# # ✅ Define image size & class labels
# IMG_SIZE = (224, 224)
# class_labels = ['10', '100', '20', '200', '2000', '500']  # Ensure this matches training class names

# # ✅ Function to preprocess input image
# def preprocess_image(image_path):
#     img = cv2.imread(image_path)  # Read the image
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     img = cv2.resize(img, IMG_SIZE)  # Resize to match model input
#     img = img / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
#     return img

# # ✅ Function to predict currency value
# def predict_currency(image_path):
#     img = preprocess_image(image_path)  # Preprocess image
#     prediction = model.predict(img)  # Get model prediction
#     predicted_class = np.argmax(prediction)  # Get highest probability index
#     currency_value = class_labels[predicted_class]  # Get currency label
#     confidence = np.max(prediction) * 100  # Confidence percentage

#     print(f"💰 Predicted Currency: {currency_value} INR")
#     print(f"🔹 Confidence: {confidence:.2f}%")

# image_path = "Twenty_rupee.jpg"  
# predict_currency(image_path)

##################################### Version - 4 ##############################################################

# import tensorflow as tf
# import numpy as np
# import cv2

# # ✅ Load the trained model
# model = tf.keras.models.load_model("currency_model.keras")
# print("✅ Model Loaded Successfully!")

# # ✅ Define image size & class labels
# IMG_SIZE = (224, 224)
# class_labels = ['10', '100', '20', '200', '2000', '50', '500']  # Ensure this matches training class names

# # ✅ Function to preprocess input image
# def preprocess_image(image):
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     img = cv2.resize(img, IMG_SIZE)  # Resize to match model input
#     img = img / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
#     return img

# # ✅ Function to predict currency value
# def predict_currency(frame):
#     img = preprocess_image(frame)  # Preprocess image
#     prediction = model.predict(img)  # Get model prediction
#     predicted_class = np.argmax(prediction)  # Get highest probability index
#     currency_value = class_labels[predicted_class]  # Get currency label
#     confidence = np.max(prediction) * 100  # Confidence percentage

#     return currency_value, confidence

# # ✅ Open Camera
# cap = cv2.VideoCapture(0)  # Use default webcam (change to 1 if using external camera)
# print("📸 Camera Opened - Press 'SPACE' to capture, 'ESC' to exit")

# while True:
#     ret, frame = cap.read()  # Capture frame
#     if not ret:
#         print("❌ Failed to grab frame")
#         break

#     cv2.imshow("Currency Detector - Press SPACE to Capture", frame)  # Show live video

#     key = cv2.waitKey(1) & 0xFF
#     if key == 32:  # Press SPACE to capture
#         print("📷 Image Captured - Processing...")
#         currency, confidence = predict_currency(frame)  # Predict currency
#         print(f"💰 Predicted Currency: {currency} INR | 🔹 Confidence: {confidence:.2f}%")

#         # Display the result on the frame
#         cv2.putText(frame, f"{currency} INR ({confidence:.2f}%)",
#                     (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         cv2.imshow("Prediction", frame)  # Show result
#         cv2.waitKey(2000)  # Wait 2 seconds before continuing

#     elif key == 27:  # Press ESC to exit
#         print("🔴 Exiting...")
#         break

# cap.release()  # Release camera
# cv2.destroyAllWindows()  # Close all windows

############################################## version - 5 #####################################################

import tensorflow as tf
import numpy as np
import cv2
import pyttsx3  # Import pyttsx3 for voice feedback

# ✅ Load the trained model
model = tf.keras.models.load_model("currency_model.keras")
print("✅ Model Loaded Successfully!")

# ✅ Define image size & class labels
IMG_SIZE = (224, 224)
class_labels = ['10', '100', '20', '200', '2000', '50', '500']  # Ensure this matches training class names

# ✅ Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 100)  # Set speed of speech
engine.setProperty('volume', 1)  # Set volume level

# ✅ Function to speak the result
def speak_result(currency):
    speech_text = f"The predicted currency is {currency} rupees "
    engine.say(speech_text)
    engine.runAndWait()

# ✅ Function to preprocess input image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, IMG_SIZE)  # Resize to match model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    return img

# ✅ Function to predict currency value
def predict_currency(frame):
    img = preprocess_image(frame)  # Preprocess image
    prediction = model.predict(img)  # Get model prediction
    predicted_class = np.argmax(prediction)  # Get highest probability index
    currency_value = class_labels[predicted_class]  # Get currency label
    confidence = np.max(prediction) * 100  # Confidence percentage

    return currency_value, confidence

# ✅ Open Camera
cap = cv2.VideoCapture(0)  # Use default webcam (change to 1 if using external camera)
print("📸 Camera Opened - Press 'SPACE' to capture, 'ESC' to exit")

while True:
    ret, frame = cap.read()  # Capture frame
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Currency Detector - Press SPACE to Capture", frame)  # Show live video

    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # Press SPACE to capture
        print("📷 Image Captured - Processing...")
        currency, confidence = predict_currency(frame)  # Predict currency
        print(f"💰 Predicted Currency: {currency} INR | 🔹 Confidence: {confidence:.2f}%")

        # Display the result on the frame
        cv2.putText(frame, f"{currency} INR ({confidence:.2f}%)",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Speak the result
        speak_result(currency)

        cv2.imshow("Prediction", frame)  # Show result
        cv2.waitKey(2000)  # Wait 2 seconds before continuing

    elif key == 27:  # Press ESC to exit
        print("🔴 Exiting...")
        break

cap.release()  # Release camera
cv2.destroyAllWindows()  # Close all windows
