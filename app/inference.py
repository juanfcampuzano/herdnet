
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
from sklearn.cluster import DBSCAN

from PIL import ImageDraw

def draw_points_on_image(image, points, radius=5, color='red'):
    draw = ImageDraw.Draw(image)
    for y, x in points:
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, fill=color, width=2)
    return image

def cluster_points(coords, eps=30, min_samples=1):
    if len(coords) == 0:
        return []

    coords_np = np.array(coords)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_np)

    clustered_coords = []
    labels = clustering.labels_

    for label in set(labels):
        if label == -1:
            continue

        cluster_points = coords_np[labels == label]
        centroid = cluster_points.mean(axis=0)
        clustered_coords.append(tuple(centroid.astype(int)))

    return clustered_coords

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
    1: 'topi',
    2: 'buffalo',
    3: 'kob',
    4: 'elephant',
    5: 'warthog',
    6: 'waterbuck'
}

def render_overlay(image_path, prediction_tensor, threshold=0.4):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    img_np = np.array(image)

    pred = prediction_tensor.squeeze().cpu().numpy()
    if pred.ndim == 2:
        pred = np.expand_dims(pred, axis=0)

    class_preds = pred[1:7]
    tmp_dir = "app/static/overlays"
    os.makedirs(tmp_dir, exist_ok=True)

    max_scores = [class_preds[i].max() for i in range(class_preds.shape[0])]
    best_idx = int(np.argmax(max_scores))
    predicted_class = best_idx + 1
    best_score = max_scores[best_idx]

    heatmap_img = image.resize((256, 256))
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(np.array(heatmap_img))
    ax.imshow(class_preds[best_idx], cmap="jet", alpha=0.5)
    ax.axis("off")

    filename = f"{uuid.uuid4().hex}_class{predicted_class}.png"
    path = os.path.join(tmp_dir, filename)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    max_score = class_preds[best_idx].max()
    threshold = 0.8 * max_score
    coords = np.argwhere(class_preds[best_idx] > threshold)

    H, W = class_preds[best_idx].shape
    scale_y = orig_h / H
    scale_x = orig_w / W
    scaled_coords = [(int(y * scale_y), int(x * scale_x)) for y, x in coords]

    clustered_coords = cluster_points(scaled_coords, eps=40)
    points_image = draw_points_on_image(image.copy(), clustered_coords, radius=15)
    points_filename = f"{uuid.uuid4().hex}_points.png"
    points_path = os.path.join(tmp_dir, points_filename)
    points_image.save(points_path)

    detected = [{
        "class_id": predicted_class,
        "name": CLASSES[predicted_class],
        "score": float(best_score),
        "overlays": {
            "heatmap": f"/static/overlays/{filename}",
            "points": f"/static/overlays/{points_filename}"
        }
    }]

    return JSONResponse(content={"species_detected": detected})



