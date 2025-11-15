import pandas as pd
from src.yolo_converter import convert_df_to_yolo

# Load the parsed CSV from EDA step
df = pd.read_csv("EDA/eda_outputs/tables/parsed_annotations.csv")

# Split
df_train = df[df["split"] == "train"]
df_val   = df[df["split"] == "val"]

# Convert
convert_df_to_yolo(
    df_train,
    images_dir="data/images/train/",
    labels_dir="yolo_dataset/labels/train/"
)

convert_df_to_yolo(
    df_val,
    images_dir="data/images/val/",
    labels_dir="yolo_dataset/labels/val/"
)