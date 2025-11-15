import os
import pandas as pd
from tqdm import tqdm

# IMPORTANT — mapping must match the YOLO class order
CLASSES = [
    "car",
    "traffic sign",
    "traffic light",
    "person",
    "truck",
    "bus",
    "bike",
    "rider",
    "motor",
    "train"
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

def convert_df_to_yolo(df, images_dir, labels_dir):
    """
    Convert parsed BDD DF into YOLO text files.
    """
    os.makedirs(labels_dir, exist_ok=True)
    
    grouped = df.groupby("image")
    
    for image_name, group in tqdm(grouped, desc="Converting to YOLO"):
        img_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, image_name.replace(".jpg", ".txt"))
        
        # image resolution (BDD100K is 1280x720)
        W, H = 1280, 720
        
        yolo_lines = []
        
        for _, row in group.iterrows():
            cls = row["category"]
            if cls not in CLASS_TO_ID:
                continue
            
            cls_id = CLASS_TO_ID[cls]
            
            # bbox coords
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            xc = (x1 + (x2 - x1) / 2) / W
            yc = (y1 + (y2 - y1) / 2) / H
            
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
