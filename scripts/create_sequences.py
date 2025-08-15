import numpy as np
import os
import glob

preprocessed_dir = 'data/processed/preprocessed/'
sequence_dir = 'data/processed/sequences/'
os.makedirs(sequence_dir, exist_ok=True)

images = sorted(glob.glob(os.path.join(preprocessed_dir, '*.npy')))

input_length = 6  # 30 minutes (6 × ~5 min intervals)
output_length = 12  # 1 hour (12 × ~5 min intervals)
for i in range(len(images) - (input_length + output_length) + 1):
    input_seq = np.array([np.load(images[j]) for j in range(i, i + input_length)])
    output_seq = np.array([np.load(images[j]) for j in range(i + input_length, i + input_length + output_length)])
    input_seq = input_seq[..., np.newaxis]
    output_seq = output_seq[..., np.newaxis]
    np.save(os.path.join(sequence_dir, f'seq_{i}_input.npy'), input_seq)
    np.save(os.path.join(sequence_dir, f'seq_{i}_output.npy'), output_seq)
    print(f"Created sequence {i}")