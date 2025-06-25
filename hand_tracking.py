import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils
# Create CSV file if it doesn't exist
if not os.path.exists('gesture_data.csv'):
    with open('gesture_data.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + ['label']
        csv_writer.writerow(header)

# Ask user for label
gesture_label = input("Enter gesture label (e.g., palm, fist, thumbs_up): ")
# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the image (optional) for selfie view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(rgb)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract x and y coordinates
            x_data = [lm.x for lm in hand_landmarks.landmark]
            y_data = [lm.y for lm in hand_landmarks.landmark]
            data_row = x_data + y_data + [gesture_label]

            # Save to CSV
            with open('gesture_data.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(data_row)
    # Display
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
