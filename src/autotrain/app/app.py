import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from autotrain import __version__, logger
from autotrain.app.api_routes import api_router
from autotrain.app.oauth import attach_oauth
from autotrain.app.ui_routes import ui_router


logger.info("Starting AutoTrain...")

app = FastAPI()
if "SPACE_ID" in os.environ:
    attach_oauth(app)

app.include_router(ui_router, prefix="/ui")
app.include_router(api_router, prefix="/api")
logger.info(f"AutoTrain version: {__version__}")
logger.info("AutoTrain started successfully")


@app.get("/")
async def forward_to_ui():
    return RedirectResponse(url="/ui/")
