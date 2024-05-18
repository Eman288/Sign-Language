import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

model_save_path = 'Sign\The saved model\model.keras'
model = tf.keras.models.load_model(model_save_path)
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2BGRA)
    # Process frame with MediaPipe Hand Gesture Recognition
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            # Preprocess landmarks (reshape, convert to numpy array, etc.)
            # Example:
            landmarks_array = np.array(landmarks, dtype=np.float32)
            landmarks_array = landmarks_array[np.newaxis, ...]  # Add batch dimension

           
            
            # Use the model to make predictions
            predictions = model.predict(landmarks_array)
            predicted_class = "nothing"
            # Example: Print the predicted class
            min_value = 0.9
            if np.max(predictions) >= min_value:
               predicted_class = ""
            # Example: Print the predicted class
            if (np.argmax(predictions) == 0):
              predicted_class = "Busy"
            elif (np.argmax(predictions) == 1):
              predicted_class ="Chair"
            elif (np.argmax(predictions) == 2):
              predicted_class ="Excuse me"
            elif (np.argmax(predictions) == 3):
              predicted_class ="Goodbye"
            elif (np.argmax(predictions) == 4):
              predicted_class ="I am fine"
            elif (np.argmax(predictions) == 5):
              predicted_class ="I am tired"
            elif (np.argmax(predictions) == 6):
              predicted_class ="I don't know"
            elif (np.argmax(predictions) == 7):
              predicted_class ="Internet"
            elif (np.argmax(predictions) == 8):
              predicted_class ="Take care"
            elif (np.argmax(predictions) == 9):
              predicted_class ="Who"
            else:
              predicted_class ="nothing"

            cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Sign App',frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
