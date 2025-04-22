import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import pandas as pd
import os

def inference_from_image(model, image_path, patch_size=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
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

    return output

def inference_from_csv(model, csv_file, root_dir, patch_size=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
    df = pd.read_csv(csv_file)
    
    all_outputs = []
    image_names = []

    for idx, row in df.iterrows():
        image_name = row['filename']
        image_path = os.path.join(root_dir, image_name)

        output = inference_from_image(model, image_path, patch_size=patch_size, device=device)
        all_outputs.append(output.cpu().numpy().tolist())  # convertir a lista para JSON
        image_names.append(image_name)

    return list(zip(image_names, all_outputs))


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from fastapi.responses import StreamingResponse

def render_overlay(image_path, prediction_tensor):
    # Abrir imagen y redimensionar
    image = Image.open(image_path).convert("RGB").resize((256, 256))
    img_np = np.array(image)

    # Obtener predicción como heatmap (shape: [1, 1, 256, 256])
    heatmap = prediction_tensor.squeeze().cpu().numpy()

    # Superponer
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Aquí puedes usar 'inferno', 'hot', etc.
    plt.axis("off")

    # Guardar imagen como bytes en buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
