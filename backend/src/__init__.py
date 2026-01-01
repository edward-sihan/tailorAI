from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from src.middleware import middleware
from src.poseDetection.routes import pose_detection_router
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import os
import mediapipe as mp

version = "v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    landmarker = None
    try:
        print("Server is starting")

        model_path = "mediaPipe_model/pose_landmarker_full.task"
        print(f"Model exists: {os.path.exists(model_path)}")

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
        )

        landmarker = PoseLandmarker.create_from_options(options)
        print("Landmarker created successfully!")

        app.state.landmarker = landmarker  # 👈 store for routes

        yield

    finally:
        if landmarker:
            landmarker.close()
            print("Landmarker closed")

        print("Server is shutting down")


app = FastAPI(
    version=version,
    title="AI Powered Tailor Shop",
    description="Backend for the AI Powered Tailoe Shop",
    lifespan=lifespan,
    docs_url=f"/api/{version}/docs",
    redoc_url=f"/api/{version}/redoc",
    contact={"name": "Yuhas"},
)

middleware(app)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    headers = getattr(exc, "headers", None)
    message = ""
    resolution = ""

    if isinstance(exc.detail, dict):
        message = exc.detail.get("message", "Unknown error")
        resolution = exc.detail.get("resolution", None)
    else:
        message = exc.detail
        resolution = None

    body = {
        "success": False,
        "data": None,
        "message": message,
        "resolution": resolution,
    }
    return JSONResponse(status_code=exc.status_code, content=body, headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "data": None,
            "message": f"Error_Type:{exc.errors()[0]['type']} -> Error_Msg:{exc.errors()[0]['msg']} -> Error_Loc:{exc.errors()[0]['loc'][0]}",
            "resolution": "Validation Failed please resolove the sent data",  # or format it however you want
        },
    )


app.include_router(
    pose_detection_router,
    prefix=f"/api/{version}/posedetection",
    tags=["pose_detection"],
)
