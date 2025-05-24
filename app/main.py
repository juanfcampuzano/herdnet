import io, os, torch, requests, tempfile, csv, zipfile
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from animaloc.models import HerdNet
from animaloc.eval.stitchers import HerdNetStitcher
from animaloc.data import ImageToPatches
from .inference import inference_from_image, inference_from_csv, render_overlay
import torch.nn as nn
import torch.nn.functional as F

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out, out
MODEL_URL = "https://proyecto-clase-despliegues.s3.us-east-2.amazonaws.com/best_model_maquina_uniandes_50epochs_86_f1.pth"
DOWN_RATIO = 2
NUM_CLASSES = 7
PATCH_SIZE = 512
DEVICE = torch.device("cpu")

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

model = None
stitcher = None

PREDICTION_LOG_PATH = os.path.join(os.path.dirname(__file__), "prediction_log.csv")
CLASSES = {
    1: 'topi', 2: 'buffalo', 3: 'kob',
    4: 'elephant', 5: 'warthog', 6: 'waterbuck'
}

def log_prediction(class_id, class_name):
    is_new = not os.path.exists(PREDICTION_LOG_PATH)
    with open(PREDICTION_LOG_PATH, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "class_id", "class_name"])
        writer.writerow([datetime.now().isoformat(), class_id, class_name])

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
    global model, stitcher
    model_path = download_model()
    base_model = HerdNet(num_classes=NUM_CLASSES, down_ratio=DOWN_RATIO)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    base_model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    base_model.to(DEVICE).eval()

    model = base_model

    wrapped_model = ModelWrapper(base_model)
    stitcher = HerdNetStitcher(
        model=wrapped_model,
        size=(PATCH_SIZE, PATCH_SIZE),
        overlap=0,
        down_ratio=DOWN_RATIO,
        reduction="mean",
        device_name="cpu"
    )

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    output = inference_from_image(model, tmp_path, device=DEVICE, stitcher=stitcher)
    pred_np = output.squeeze().cpu().numpy()
    if pred_np.ndim == 2:
        pred_np = pred_np[None]

    best_idx = int(pred_np[1:7].max(axis=(1,2)).argmax()) + 1
    log_prediction(best_idx, CLASSES[best_idx])
    return {"shape": list(output.shape), "output": pred_np.tolist()}

@app.post("/predict/image/overlay")
async def predict_image_overlay(file: UploadFile = File(...)):
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    output = inference_from_image(model, tmp_path, device=DEVICE, stitcher=stitcher)
    pred_np = output.squeeze().cpu().numpy()
    if pred_np.ndim == 2:
        pred_np = pred_np[None]

    best_idx = int(pred_np[1:7].max(axis=(1,2)).argmax()) + 1
    log_prediction(best_idx, CLASSES[best_idx])
    return render_overlay(tmp_path, output)

@app.get("/")
async def root():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/dashboard")
async def dashboard():
    with open("app/templates/dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/dashboard/data")
async def dashboard_data():
    data = []
    if os.path.exists(PREDICTION_LOG_PATH):
        with open(PREDICTION_LOG_PATH, newline='', encoding="utf-8") as f:
            data = list(csv.DictReader(f))
    return JSONResponse(content={"history": data})
