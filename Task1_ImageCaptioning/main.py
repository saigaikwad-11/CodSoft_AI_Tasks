import io
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

app = FastAPI()
templates = Jinja2Templates(directory="app")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Model hata diya hai taaki error na aaye aur sirf interface test ho sake
  return {"caption": "A group of kids playing football on the green field!"}