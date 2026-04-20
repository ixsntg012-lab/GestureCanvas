import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        hand_landmarks = []

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS
                )

                lm_list = []
                for id, lm in enumerate(hand.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                hand_landmarks.append(lm_list)

        return frame, hand_landmarks