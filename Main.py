import cv2
import mediapipe as mp
import pyautogui

# Initialize webcam, Face Mesh, and Hands
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
hands = mp.solutions.hands.Hands()
screen_w, screen_h = pyautogui.size()

# Scroll control variables
previous_y = None
scroll_threshold = 20  # Adjust this value for sensitivity

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    hand_output = hands.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Calculate the position of the mouse pointer using the right eye
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                pyautogui.moveTo(screen_x, screen_y)

        # Detecting blink for clicking (right eye)
        right_eye = [landmarks[145], landmarks[159]]
        for landmark in right_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        if (right_eye[0].y - right_eye[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    # Hand gesture detection for scrolling
    if hand_output.multi_hand_landmarks:
        for hand_landmarks in hand_output.multi_hand_landmarks:
            # Use the index finger tip for scrolling
            index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_tip.x * frame_w)
            y = int(index_finger_tip.y * frame_h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            if previous_y is not None:
                delta_y = y - previous_y
                if abs(delta_y) > scroll_threshold:
                    if delta_y > 0:
                        pyautogui.scroll(-3)  # Scroll down
                        print("Scrolling down")
                    else:
                        pyautogui.scroll(3)  # Scroll up
                        print("Scrolling up")
                print(f"Delta Y: {delta_y}")
            previous_y = y

    cv2.imshow('Eye and Hand Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cam.release()
cv2.destroyAllWindows()
