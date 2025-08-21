import numpy as np
import glob
import os
import cv2
from PIL import Image

# Load and preprocess Mumbai radar frames from hard disk
frame_dir = "/media/gayatri/Elements/mumbai/raw/"
frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
if len(frame_files) < 12:
    print("Need at least 12 frames for training sequences.")
    exit()

frames = [np.array(Image.open(f).convert("L")) for f in frame_files]  # Grayscale
frames = [cv2.resize(f, (128, 128)) / 255.0 for f in frames]  # Resize and normalize
frames = np.array(frames)

# Create overlapping sequences (6 frames input, 6 frames target)
sequence_length = 6
sequences_x, sequences_y = [], []
for i in range(len(frames) - sequence_length * 2 + 1):
    seq_x = frames[i:i + sequence_length]
    seq_y = frames[i + sequence_length:i + sequence_length * 2]
    sequences_x.append(seq_x)
    sequences_y.append(seq_y)

sequences_x = np.array(sequences_x)[..., np.newaxis]  # Shape: (n, 6, 128, 128, 1)
sequences_y = np.array(sequences_y)[..., np.newaxis]  # Shape: (n, 6, 128, 128, 1)


# Save to laptop's data directory
os.makedirs("data/mumbai/processed/sequences/", exist_ok=True)
np.save("data/mumbai/processed/sequences/train_x.npy", sequences_x)
np.save("data/mumbai/processed/sequences/train_y.npy", sequences_y)
print(f"Training data saved: {sequences_x.shape} input, {sequences_y.shape} target")