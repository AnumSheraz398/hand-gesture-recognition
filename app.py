import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

# Load trained model and label encoder
model = tf.keras.models.load_model('gesture_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

print("👋 Starting webcam. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for natural (mirror) view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract x and y coordinates
            landmarks = hand_landmarks.landmark
            x = [lm.x for lm in landmarks]
            y = [lm.y for lm in landmarks]

            # Combine x and y into one feature vector
            features = np.array(x + y).reshape(1, -1)

            prediction = model.predict(features)[0]  # Get the 1D array of probabilities
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class] * 100  # Convert to percentage
            label = label_encoder.inverse_transform([predicted_class])[0]

            # Show label
            cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and close
cap.release()
cv2.destroyAllWindows()
