from typing import List

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger


app = FastAPI()
app.mount("/css", StaticFiles(directory="css"), name="css")  # Mounting the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")  # Mounting the static directory
templates = Jinja2Templates(directory="templates")  # Assuming your HTML is in a folder named 'templates'


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """
    This function is used to render the HTML file
    :param request:
    :return:
    """
    return templates.TemplateResponse("index.html", {"request": request})  # The form.html is your saved html file


@app.get("/params/{task}", response_class=JSONResponse)
async def fetch_params(task: str):
    """
    This function is used to fetch the parameters for a given task
    :param task: str
    :return: JSONResponse
    """
    logger.info(f"Task: {task}")
    ret = {f"{task}_arg{j}": f"value{j}" for j in range(1, 25)}
    return ret


@app.post("/create_project", response_class=JSONResponse)
async def handle_form(
    project_name: str = Form(...),
    task: str = Form(...),
    token: str = Form(...),
    username: str = Form(...),
    hardware: str = Form(...),
    data_files: List[UploadFile] = File(...),
    params: str = Form(...),
):
    """
    This function is used to handle the form submission
    :param request:
    :param project_name:
    :param task:
    :param token:
    :return:
    """
    data = {
        "project_name": project_name,
        "token": token,
        "task": task,
        "username": username,
        "hardware": hardware,
        "data_files": data_files,
        "params": params,
    }
    logger.info(data)
    return {"success": "true"}
