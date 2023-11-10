from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()
app.mount("/css", StaticFiles(directory="css"), name="css")  # Mounting the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")  # Mounting the static directory
templates = Jinja2Templates(directory="templates")  # Assuming your HTML is in a folder named 'templates'


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})  # The form.html is your saved html file


@app.get("/llm", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("llm.html", {"request": request})  # The form.html is your saved html file


@app.post("/create_project")
async def handle_form(request: Request, project_name: str = Form(...), task: str = Form(...), token: str = Form(...)):
    data = {"project_name": project_name, "token": token, "task": task}
    return templates.TemplateResponse("submission_response.html", {"request": request, "data": data})
