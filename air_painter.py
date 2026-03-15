import cv2
import numpy as np
import mediapipe as mp # Hand-tracking library

# Helper functions
def fingers_up(hand_landmarks, handedness_label="Right"):
    """
    Returns a list of 5 ints (thumb, index, middle, ring, pinky)
    where 1 means finger is up and 0 means down.
    """
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]

    lm = hand_landmarks.landmark
    fingers = []

    # Thumb: compare x depending on left/right hand
    if handedness_label == "Right":
        fingers.append(1 if lm[tips[0]].x < lm[pips[0]].x else 0)
    else:
        fingers.append(1 if lm[tips[0]].x > lm[pips[0]].x else 0)

    # Other 4 fingers: tip above PIP joint means finger up
    for i in range(1, 5):
        fingers.append(1 if lm[tips[i]].y < lm[pips[i]].y else 0)

    return fingers


def draw_ui(img, current_color):
    """
    Draw top UI buttons on the live frame.
    """
    # Clear button
    cv2.rectangle(img, (20, 10), (120, 70), (200, 200, 200), -1)
    cv2.rectangle(img, (20, 10), (120, 70), (50, 50, 50), 2) # Border
    cv2.putText(img, "CLEAR", (35, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Color buttons
    button_colors = [
        ("BLUE", (255, 0, 0), (140, 10), (240, 70)),
        ("GREEN", (0, 255, 0), (260, 10), (360, 70)),
        ("RED", (0, 0, 255), (380, 10), (480, 70)),
        ("YELLOW", (0, 255, 255), (500, 10), (620, 70)),
    ]

    for name, color, pt1, pt2 in button_colors: # Draw buttons
        cv2.rectangle(img, pt1, pt2, color, -1)
        thickness = 4 if current_color == color else 2
        cv2.rectangle(img, pt1, pt2, (255, 255, 255), thickness)
        text_color = (255, 255, 255) if name != "YELLOW" else (80, 80, 80)
        cv2.putText(img, name, (pt1[0] + 10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

    return img


# Basically check if fingertip is touching a button
def select_button(x, y):
    """
    Returns action based on fingertip position over top UI.
    """
    if 20 <= x <= 120 and 10 <= y <= 70:
        return "CLEAR"
    elif 140 <= x <= 240 and 10 <= y <= 70:
        return (255, 0, 0)   # BLUE
    elif 260 <= x <= 360 and 10 <= y <= 70:
        return (0, 255, 0)   # GREEN
    elif 380 <= x <= 480 and 10 <= y <= 70:
        return (0, 0, 255)   # RED
    elif 500 <= x <= 620 and 10 <= y <= 70:
        return (0, 255, 255) # YELLOW
    return None



# Main
cap = cv2.VideoCapture(0)

# setting a resolution
cap.set(3, 1280) # Width
cap.set(4, 720) # Height

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils # Draw hand skeleton

hands = mp_hands.Hands(
    static_image_mode=False, # Use video tracking mode
    max_num_hands=1, # Detect only one hand
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Canvas for drawing
canvas = None

# Current drawing color
current_color = (255, 0, 0)  # Blue

# Previous point for drawing continuous lines
prev_x, prev_y = 0, 0

# Brush thickness
brush_thickness = 6

while True:
    success, frame = cap.read() # Read frame
    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe requires RGB
    results = hands.process(rgb)

    frame = draw_ui(frame, current_color) # Draw buttons on screen

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Handedness label if available
        handedness_label = "Right"
        if results.multi_handedness:
            handedness_label = results.multi_handedness[0].classification[0].label # Returns left or right

        h, w, _ = frame.shape
        lm = hand_landmarks.landmark

        # Index fingertip coordinates
        x1, y1 = int(lm[8].x * w), int(lm[8].y * h)

        # Middle fingertip coordinates
        x2, y2 = int(lm[12].x * w), int(lm[12].y * h)

        # Determine which fingers are up
        up = fingers_up(hand_landmarks, handedness_label)
        thumb_up, index_up, middle_up, ring_up, pinky_up = up

        # Draw landmarks for visualization: HAND SKELETON
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Selection mode: index + middle up
        if index_up and middle_up:
            prev_x, prev_y = 0, 0 # Reset drawing

            cv2.circle(frame, (x1, y1), 12, (255, 255, 255), -1)
            cv2.circle(frame, (x2, y2), 12, (255, 255, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

            # If index fingertip is over top menu, select button
            action = select_button(x1, y1)
            if action == "CLEAR":
                canvas[:] = 0
            elif isinstance(action, tuple):
                current_color = action

        # Drawing mode: index up, middle down
        elif index_up and not middle_up:
            cv2.circle(frame, (x1, y1), 10, current_color, -1)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x1, y1

            cv2.line(frame, (prev_x, prev_y), (x1, y1), current_color, brush_thickness)
            cv2.line(canvas, (prev_x, prev_y), (x1, y1), current_color, brush_thickness)

            prev_x, prev_y = x1, y1

        else:
            prev_x, prev_y = 0, 0 # Reset if not fingers detected (STOPS DRAWING)

    else:
        prev_x, prev_y = 0, 0

    # Combine live frame and persistent canvas
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV) # CREATE INVERSE MASK (White everywhere other than canvas)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Virtual Painter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()