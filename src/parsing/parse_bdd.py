import json
import os
import pandas as pd
from tqdm import tqdm

DETECTION_CLASSES = {
    "bike", "bus", "car", "motor", "person",
    "rider", "traffic light", "traffic sign", "train", "truck"
}

def parse_bdd_labels(json_path: str, split: str) -> pd.DataFrame:
    """
    Parse BDD100K detection labels JSON into a structured pandas DataFrame.

    Args:
        json_path (str): Path to JSON annotation file.
        split (str): "train" or "val" (stored as metadata).

    Returns:
        pd.DataFrame: One row per labeled object, including metadata attributes.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for item in tqdm(data, desc=f"Parsing {split} split"):
        img = item["name"]
        attr = item.get("attributes", {})
        weather = attr.get("weather", "unknown")
        scene = attr.get("scene", "unknown")
        tod = attr.get("timeofday", "unknown")

        for label in item.get("labels", []):
            category = label.get("category")
            if category not in DETECTION_CLASSES:
                continue

            bbox = label.get("box2d")
            if bbox is None:
                continue

            rows.append({
                "image": img,
                "split": split,
                "category": category,
                "x1": float(bbox["x1"]),
                "y1": float(bbox["y1"]),
                "x2": float(bbox["x2"]),
                "y2": float(bbox["y2"]),
                "width": float(bbox["x2"] - bbox["x1"]),
                "height": float(bbox["y2"] - bbox["y1"]),
                "aspect_ratio": float((bbox["x2"] - bbox["x1"]) / (bbox["y2"] - bbox["y1"] + 1e-6)),
                "occluded": label.get("attributes", {}).get("occluded", False),
                "truncated": label.get("attributes", {}).get("truncated", False),
                "weather": weather,
                "scene": scene,
                "timeofday": tod,
            })

    return pd.DataFrame(rows)