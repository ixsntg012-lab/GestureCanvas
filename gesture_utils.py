import math

def find_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)


def is_pinch(hand_landmarks):
    # Thumb tip = 4, Index tip = 8
    thumb = hand_landmarks[4]
    index = hand_landmarks[8]

    distance = find_distance(thumb, index)

    if distance < 40:  # threshold
        return True
    return False