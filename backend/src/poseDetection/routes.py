from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from src.poseDetection.dependencies import get_landmarker
from src.poseDetection.utils import get_landmark_distance
import mediapipe as mp
from PIL import Image
import numpy as np
import io


pose_detection_router = APIRouter()


@pose_detection_router.post("/")
async def get_pose_detection(
    image: UploadFile = File(...), landmarker=Depends(get_landmarker)
):
    try:
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid Image type",
                    "resolution": "Please send a valid image of type .png or .jpg",
                },
            )

        image_bytes = await image.read()

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(pil_image)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

        landmarker_result = landmarker.detect(mp_image)

        world_landmarks = landmarker_result.pose_world_landmarks[0]

        world_landmarks_distance = get_landmark_distance(world_landmarks)

        return {
            "message": "Successfully retrieved the Tailor measurements",
            "data": world_landmarks_distance,
            "success": True,
            "resolution": None,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": str(e),
                "resolution": "Something went wrong in the get_pose_detection router. Please Try Again",
            },
        )
