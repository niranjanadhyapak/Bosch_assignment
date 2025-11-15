import os

# Absolute path to your dataset directory
DATA_DIR = r"C:\Users\niran\OneDrive\Desktop\Bosch\Assignment\EDA\data"

IMAGE_DIR = os.path.join(DATA_DIR, "images")
LABEL_DIR = os.path.join(DATA_DIR, "labels")

TRAIN_LABELS = os.path.join(LABEL_DIR, "bdd100k_labels_images_train.json")
VAL_LABELS = os.path.join(LABEL_DIR, "bdd100k_labels_images_val.json")