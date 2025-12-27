from fastapi import FastAPI, exceptions
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
# import cv2


app = FastAPI()


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points"""
    x_diff = point2.x - point1.x
    y_diff = point2.y - point1.y
    z_diff = point2.z - point1.z

    distance = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
    return distance


def extract_measurements(landmarks):
    """Extract key measurements from landmarks"""

    # Landmark indices from MediaPipe
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    NOSE = 0

    measurements = {}

    # Check if landmarks are detected
    if not landmarks or len(landmarks) == 0:
        return None

    try:
        # Shoulder width (distance between left and right shoulder)
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)
        measurements["shoulder_width_m"] = shoulder_width
        measurements["shoulder_width_cm"] = shoulder_width * 100

        # Right sleeve length (shoulder to wrist)
        right_shoulder = landmarks[RIGHT_SHOULDER]
        right_wrist = landmarks[RIGHT_WRIST]
        right_sleeve = calculate_distance(right_shoulder, right_wrist)
        measurements["right_sleeve_length_m"] = right_sleeve
        measurements["right_sleeve_length_cm"] = right_sleeve * 100

        # Left sleeve length
        left_shoulder = landmarks[LEFT_SHOULDER]
        left_wrist = landmarks[LEFT_WRIST]
        left_sleeve = calculate_distance(left_shoulder, left_wrist)
        measurements["left_sleeve_length_m"] = left_sleeve
        measurements["left_sleeve_length_cm"] = left_sleeve * 100

        # Hip width
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        hip_width = calculate_distance(left_hip, right_hip)
        measurements["hip_width_m"] = hip_width
        measurements["hip_width_cm"] = hip_width * 100

        # Torso length (shoulder to hip)
        right_shoulder = landmarks[RIGHT_SHOULDER]
        right_hip = landmarks[RIGHT_HIP]
        torso_length = calculate_distance(right_shoulder, right_hip)
        measurements["torso_length_m"] = torso_length
        measurements["torso_length_cm"] = torso_length * 100

        # Arm length (shoulder to elbow)
        right_shoulder = landmarks[RIGHT_SHOULDER]
        right_elbow = landmarks[RIGHT_ELBOW]
        arm_length = calculate_distance(right_shoulder, right_elbow)
        measurements["right_arm_length_m"] = arm_length
        measurements["right_arm_length_cm"] = arm_length * 100

        # Height (nose to ankle)
        nose = landmarks[NOSE]
        right_ankle = landmarks[RIGHT_ANKLE]
        height = calculate_distance(nose, right_ankle)
        measurements["height_m"] = height
        measurements["height_cm"] = height * 100

        return measurements

    except Exception as e:
        return None


# model_path = "../model/pose_landmarker_full.task"
# BaseOptions = mp.tasks.BaseOptions
# PoseLandmarker = mp.tasks.vision.PoseLandmarker
# PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode
#
# options = PoseLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.IMAGE,
# )
# mp_image = mp.Image.create_from_file("../image/example.png")
# landmarker = PoseLandmarker.create_from_options(options)


print("Starting MediaPipe initialization...")

# Check if files exist first
model_path = "model/pose_landmarker_full.task"
image_path = "image/IMG_5837.JPG"
# image_path = "image/example.png"


print(f"Model exists: {os.path.exists(model_path)}")

print(f"Checking image file: {image_path}")
print(f"Image exists: {os.path.exists(image_path)}")

try:
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    print("Creating options...")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )

    print("Creating landmarker...")
    landmarker = PoseLandmarker.create_from_options(options)
    print("Landmarker created successfully!")

    print("Loading image...")
    mp_image = mp.Image.create_from_file(image_path)
    print("Image loaded successfully!")

except Exception as e:
    print(f"Error during initialization: {e}")
    import traceback

    traceback.print_exc()


@app.get("/")
async def root():
    """Now the endpoint just uses the already-loaded resources"""

    try:
        landmarker_result = landmarker.detect(mp_image)

        if not landmarker_result.pose_world_landmarks:
            return {"error": "No world landmarks detected"}

        # Use world landmarks (these are in meters)
        landmarks = landmarker_result.pose_world_landmarks[0]
        # landmarks = landmarker_result.pose_landmarks[0]

        # Landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24

        def calculate_distance(point1, point2):
            import math

            x_diff = point2.x - point1.x
            y_diff = point2.y - point1.y
            z_diff = point2.z - point1.z
            return math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

        # Calculate measurements
        shoulder_width = calculate_distance(
            landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
        )

        right_sleeve = calculate_distance(
            landmarks[RIGHT_SHOULDER], landmarks[RIGHT_WRIST]
        )

        hip_width = calculate_distance(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])

        return {
            "status": "success",
            # "shoulder_width_cm": round(shoulder_width * 100, 2),
            "shoulder_width_cm": shoulder_width,
            # "right_sleeve_length_cm": round(right_sleeve * 100, 2),
            "right_sleeve_length_cm": right_sleeve,
            "hip_width_cm": hip_width,
        }

    except Exception as e:
        return {"error": str(e)}
