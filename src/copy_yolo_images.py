import shutil
import os
from tqdm import tqdm
def copy_images(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in tqdm(os.listdir(src_dir)):
        if fname.endswith(".jpg"):
            shutil.copy(
                os.path.join(src_dir, fname),
                os.path.join(dst_dir, fname)
            )
print(os.getcwd())
copy_images("./EDA/data/images/train/", "./EDA/yolo_dataset/images/train/")
copy_images("./EDA/data/images/val/",   "./EDA/yolo_dataset/images/val/")