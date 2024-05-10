from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from autotrain import __version__, logger
from autotrain.app.models import APICreateProjectModel
from autotrain.app.utils import token_verification


api_router = APIRouter()


"""
api elements:
- create_project
- logs
- stop_training
"""


def api_auth(request: Request):
    authorization = request.headers.get("Authorization")
    if authorization:
        schema, _, token = authorization.partition(" ")
        if schema.lower() == "bearer":
            token = token.strip()
            try:
                _ = token_verification(token=token)
                return token
            except Exception as e:
                logger.error(f"Failed to verify token: {e}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token: Bearer",
                )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
    )


@api_router.post("/create_project", response_class=JSONResponse)
async def api_create_project(project: APICreateProjectModel, authenticated: bool = Depends(api_auth)):
    """
    This function is used to create a new project
    :param project: APICreateProjectModel
    :return: JSONResponse
    """
    return {"success": "true"}


@api_router.get("/version", response_class=JSONResponse)
async def api_version():
    """
    This function is used to get the version of the API
    :return: JSONResponse
    """
    return {"version": __version__}
