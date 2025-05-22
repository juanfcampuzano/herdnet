import io
import os
import torch
import requests
import tempfile
import csv
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from animaloc.models import HerdNet
from animaloc.eval import HerdNetStitcher
from .inference import inference_from_image, inference_from_csv, render_overlay
import zipfile

MODEL_URL = "https://proyecto-clase-despliegues.s3.us-east-2.amazonaws.com/best_model_maquina_uniandes_50epochs_86_f1.pth"

DOWN_RATIO = 2
NUM_CLASSES = 7
PATCH_SIZE = 512

device = torch.device("cpu")

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

model = None
stitcher = None

PREDICTION_LOG_PATH = os.path.join(os.path.dirname(__file__), "prediction_log.csv")

try:
    from .inference import CLASSES
except ImportError:
    CLASSES = {
    0: 'topi',
    1: 'buffalo',
    2: 'kob',
    3: 'elephant',
    4: 'warthog',
    5: 'waterbuck'
}


def log_prediction(class_id: int, class_name: str):
    is_new = not os.path.exists(PREDICTION_LOG_PATH)
    with open(PREDICTION_LOG_PATH, mode="a", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if is_new:
            writer.writerow(["timestamp", "class_id", "class_name"])
        writer.writerow([
            datetime.now().isoformat(),
            class_id,
            class_name
        ])

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
    model_instance = HerdNet(num_classes=NUM_CLASSES, down_ratio=DOWN_RATIO)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model_instance.load_state_dict(cleaned_state_dict)
    model_instance.to(device)
    model_instance.eval()

    model = model_instance

    stitcher = HerdNetStitcher(
        model=model,
        size=(PATCH_SIZE, PATCH_SIZE),
        overlap=0,
        down_ratio=DOWN_RATIO,
        reduction="mean",
        device_name="cpu"
    )

    stitcher.device = device
    stitcher.model.to(device)

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    output = inference_from_image(model, tmp_path, device=device)
    output_np = output.squeeze().cpu().numpy()

    import numpy as np
    if output_np.ndim == 2:
        output_np = np.expand_dims(output_np, axis=0)
    class_preds = output_np[:6]
    max_scores = [class_preds[i].max() for i in range(class_preds.shape[0])]
    predicted_class = int(np.argmax(max_scores))
    class_name = CLASSES.get(predicted_class, str(predicted_class))
    log_prediction(predicted_class, class_name)

    return {
        "shape": list(output.shape),
        "output": output_np.tolist()
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

    results = inference_from_csv(model, csv_path, root_dir, device=device)
    return {"results": results}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    with open("app/templates/dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/dashboard/data", response_class=JSONResponse)
async def dashboard_data():
    data = []
    if os.path.exists(PREDICTION_LOG_PATH):
        with open(PREDICTION_LOG_PATH, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    return JSONResponse(content={"history": data})


@app.post("/predict/image/overlay")
async def predict_image_overlay(file: UploadFile = File(...)):
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    output = inference_from_image(model, tmp_path, device=device)
    output_np = output.squeeze().cpu().numpy()

    import numpy as np
    if output_np.ndim == 2:
        output_np = np.expand_dims(output_np, axis=0)
    class_preds = output_np[:6]
    max_scores = [class_preds[i].max() for i in range(class_preds.shape[0])]
    predicted_class = int(np.argmax(max_scores))
    class_name = CLASSES.get(predicted_class, str(predicted_class))
    log_prediction(predicted_class, class_name)

    return render_overlay(tmp_path, output)