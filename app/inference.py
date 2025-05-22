
import os
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from animaloc.utils.seed import set_seed
from fastapi.responses import JSONResponse
import uuid
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse

set_seed(9292)

PATCH_SIZE = 512

transforms = A.Compose([
    A.Resize(PATCH_SIZE, PATCH_SIZE),
    A.Normalize(p=1.0)
])

def inference_from_image(model, image_path, patch_size=PATCH_SIZE, device='cpu'):
    device = torch.device(device)
    model.eval()
    model.to(device)

    image = np.array(Image.open(image_path).convert("RGB"))

    transform = A.Compose([
        A.Resize(patch_size, patch_size),
        A.Normalize(),
        ToTensorV2()
    ])

    input_tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        return output[1] if isinstance(output, (list, tuple)) else output

def inference_from_csv(model, csv_file, root_dir, patch_size=PATCH_SIZE, device='cpu'):
    device = torch.device(device)
    df = pd.read_csv(csv_file)

    all_outputs = []
    image_names = []

    for idx, row in df.iterrows():
        image_name = row['filename']
        image_path = os.path.join(root_dir, image_name)

        output = inference_from_image(model, image_path, patch_size=patch_size, device=device)
        all_outputs.append(output.cpu().numpy().tolist())
        image_names.append(image_name)

    return list(zip(image_names, all_outputs))

CLASSES = {
    0: 'topi',
    1: 'buffalo',
    2: 'kob',
    3: 'elephant',
    4: 'warthog',
    5: 'waterbuck'
}

def render_overlay(image_path, prediction_tensor, threshold=0.4):
    image = Image.open(image_path).convert("RGB").resize((256, 256))
    img_np = np.array(image)

    pred = prediction_tensor.squeeze().cpu().numpy()
    if pred.ndim == 2:
        pred = np.expand_dims(pred, axis=0)

    class_preds = pred[:6]
    tmp_dir = "app/static/overlays"
    os.makedirs(tmp_dir, exist_ok=True)

    max_scores = [class_preds[i].max() for i in range(class_preds.shape[0])]
    best_idx = int(np.argmax(max_scores))
    best_score = max_scores[best_idx]

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_np)
    ax.imshow(class_preds[best_idx], cmap="jet", alpha=0.5)
    ax.set_title(f"{CLASSES[best_idx]} ({best_score:.2f})")
    ax.axis("off")

    filename = f"{uuid.uuid4().hex}_class{best_idx+1}.png"
    path = os.path.join(tmp_dir, filename)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    detected = [{
        "class_id": best_idx,
        "name": CLASSES[best_idx],
        "score": float(best_score),
        "image": f"/static/overlays/{filename}"
    }]

    return JSONResponse(content={"species_detected": detected})



