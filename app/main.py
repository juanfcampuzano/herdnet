import io
import os
import torch
import requests
import tempfile
from fastapi import FastAPI, UploadFile, File
from tempfile import NamedTemporaryFile
from animaloc.models import HerdNet
from .inference import inference_from_image, inference_from_csv, render_overlay
import zipfile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request

MODEL_URL = "https://proyecto-clase-despliegues.s3.us-east-2.amazonaws.com/herdnet_model_v.1.0.1.pth"
NUM_CLASSES = 7

app = FastAPI()
model = None

def download_model():
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "model.pth")
    
    with open(tmp_path, "wb") as f:
        f.write(response.content)
    
    return tmp_path

@app.on_event("startup")
def load_model():
    global model
    model_path = download_model()
    model = HerdNet(num_classes=NUM_CLASSES)

    checkpoint = torch.load(model_path, map_location="cpu")
    
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(cleaned_state_dict)
    model.eval()

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    output, *_ = inference_from_image(model, tmp_path)

    return {
        "shape": list(output.shape),
        "output": output.squeeze().cpu().numpy().tolist()
    }

@app.post("/predict/csv")
async def predict_csv(csv_file: UploadFile = File(...), images: UploadFile = File(...)):
    tmp_dir = tempfile.gettempdir()
    csv_path = os.path.join(tmp_dir, csv_file.filename)
    with open(csv_path, "wb") as f:
        f.write(await csv_file.read())

    root_dir = os.path.join(tmp_dir, "images")
    os.makedirs(root_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(await images.read())) as z:
        z.extractall(root_dir)

    results = inference_from_csv(model, csv_path, root_dir)
    return {"results": results}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
    
@app.post("/predict/image/overlay")
async def predict_image_overlay(file: UploadFile = File(...)):
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    output, *_ = inference_from_image(model, tmp_path)

    return render_overlay(tmp_path, output)
