
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
import torchvision.transforms as T

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

def inference_from_image(model, image_path, device='cpu', stitcher=None):
    device = torch.device(device)
    model.eval()
    model.to(device)

    image = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image)

    with torch.no_grad():
        output = stitcher(image_tensor)
        return output

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
    from matplotlib import pyplot as plt
    import uuid
    from PIL import ImageDraw

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    img_np = np.array(image)

    pred = prediction_tensor.squeeze().cpu().numpy()
    if pred.ndim == 2:
        pred = np.expand_dims(pred, axis=0)

    background = pred[0]
    class_preds = pred[1:7]
    tmp_dir = "app/static/overlays"
    os.makedirs(tmp_dir, exist_ok=True)

    diff_maps = class_preds - background
    best_idx = int(np.argmax([m.sum() for m in diff_maps]))
    predicted_class = best_idx + 1
    heatmap = diff_maps[best_idx]
    best_score = float(heatmap.sum())

    heatmap_img = image.resize((256, 256))
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(np.array(heatmap_img))
    ax.imshow(heatmap, cmap="jet", alpha=0.5)
    ax.axis("off")

    filename = f"{uuid.uuid4().hex}_class{predicted_class}.png"
    path = os.path.join(tmp_dir, filename)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    min_val = heatmap.min()
    thresh = min_val + 0.2 * (heatmap.max() - min_val)
    coords = np.argwhere(heatmap < thresh)

    H, W = heatmap.shape
    scale_y = orig_h / H
    scale_x = orig_w / W
    scaled_coords = [(int(y * scale_y), int(x * scale_x)) for y, x in coords]

    clustered_coords = cluster_points(scaled_coords, eps=20)
    points_image = draw_points_on_image(image.copy(), clustered_coords, radius=15)
    points_filename = f"{uuid.uuid4().hex}_points.png"
    points_path = os.path.join(tmp_dir, points_filename)
    points_image.save(points_path)

    detected = [{
        "class_id": predicted_class,
        "name": CLASSES[predicted_class],
        "score": best_score,
        "overlays": {
            "heatmap": f"/static/overlays/{filename}",
            "points": f"/static/overlays/{points_filename}"
        }
    }]

    return JSONResponse(content={"species_detected": detected})

