import math


def calculate_distance(point1, point2):
    x = (point2.x - point1.x) ** 2
    y = (point2.y - point1.y) ** 2
    z = (point2.z - point1.z) ** 2
    return math.sqrt(x + y + z) * 100  # convert to cm


def get_landmark_distance(landmarks):
    CALIBRATION_FACTOR = 1.19

    # Landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    RIGHT_ANKLE = 28

    measurements = {
        "shoulder_width_cm": calculate_distance(
            landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
        ),
        "chest_approximation_cm": calculate_distance(
            landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
        ),
        "right_sleeve_length_cm": calculate_distance(
            landmarks[RIGHT_SHOULDER], landmarks[RIGHT_WRIST]
        ),
        "left_sleeve_length_cm": calculate_distance(
            landmarks[LEFT_SHOULDER], landmarks[LEFT_WRIST]
        ),
        "right_arm_length_cm": calculate_distance(
            landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW]
        ),
        "left_arm_length_cm": calculate_distance(
            landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW]
        ),
        "torso_length_cm": calculate_distance(
            landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP]
        ),
        "hip_width_cm": calculate_distance(landmarks[LEFT_HIP], landmarks[RIGHT_HIP]),
        "inseam_length_cm": calculate_distance(
            landmarks[RIGHT_HIP], landmarks[RIGHT_ANKLE]
        ),
        "height_cm": calculate_distance(landmarks[NOSE], landmarks[RIGHT_ANKLE]),
    }

    for key in measurements:
        measurements[key] = measurements[key] * CALIBRATION_FACTOR

    return measurements
