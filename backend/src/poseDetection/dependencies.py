from fastapi import Request


async def get_landmarker(request: Request):
    return request.app.state.landmarker
