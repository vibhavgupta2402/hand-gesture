import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import time
cap = cv2.VideoCapture(0)
initHand = mp.solutions.hands
mainHand = initHand.Hands(max_num_hands=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)
draw = mp.solutions.drawing_utils
finger_tip_indices = [4, 8, 12, 16, 20]

last_action_time = time.time()
debounce_time = 2 

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mainHand.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_tip_y_coordinates = [int(hand_landmarks.landmark[i].y * frame.shape[0]) for i in finger_tip_indices]
            num_extended_fingers = sum(1 for y_coord in finger_tip_y_coordinates if y_coord < finger_tip_y_coordinates[0])
            if num_extended_fingers == 1:
                webbrowser.open("https://www.google.com")
                time.sleep(3)
            elif num_extended_fingers == 2:
                webbrowser.open("https://www.facebook.com")
                time.sleep(3)
            elif num_extended_fingers == 3:
                webbrowser.open("https://instagram.com")
                time.sleep(3)
            draw.draw_landmarks(frame, hand_landmarks, initHand.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
