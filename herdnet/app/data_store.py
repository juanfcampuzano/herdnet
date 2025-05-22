from datetime import datetime
import numpy as np

history = []

def add_prediction(filename, prediction_tensor):
    pred = prediction_tensor.squeeze().cpu().numpy()
    unique, counts = np.unique(pred, return_counts=True)
    class_counts = dict(zip(map(int, unique), map(int, counts)))

    history.append({
        "filename": filename,
        "timestamp": datetime.now().isoformat(),
        "class_counts": class_counts
    })

def get_stats():
    total = len(history)
    last = history[-1]["timestamp"] if total > 0 else "No predictions yet"
    return {
        "total_predictions": total,
        "last_prediction": last
    }

def get_history():
    return history
