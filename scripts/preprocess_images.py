import cv2
import numpy as np
import os
import glob

input_dir = 'data/processed/images/'
output_dir = 'data/processed/preprocessed/'
os.makedirs(output_dir, exist_ok=True)

images = glob.glob(os.path.join(input_dir, '*.png'))
for img_path in images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img_resized = cv2.resize(img, (128, 128))
        img_normalized = img_resized / 255.0
        output_file = os.path.join(output_dir, os.path.basename(img_path).replace('.png', '.npy'))
        np.save(output_file, img_normalized)
        print(f"Preprocessed: {output_file}")
    else:
        print(f"Failed to load: {img_path}")