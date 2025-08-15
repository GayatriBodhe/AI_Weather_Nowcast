import numpy as np
import os
import glob

preprocessed_dir = 'data/processed/preprocessed/'
sequence_dir = 'data/processed/sequences/'
os.makedirs(sequence_dir, exist_ok=True)

images = sorted(glob.glob(os.path.join(preprocessed_dir, '*.npy')))

input_length = 6
output_length = 1  # Predict next 1 frame for training

for i in range(len(images) - (input_length + output_length) + 1):
    input_seq = [np.load(images[j]) for j in range(i, i + input_length)]
    output_seq = [np.load(images[j]) for j in range(i + input_length, i + input_length + output_length)]
    input_seq = np.stack(input_seq, axis=0)[:, :, :, np.newaxis]
    output_seq = np.stack(output_seq, axis=0)[:, :, :, np.newaxis]  # Shape: (1, 128, 128, 1)
    np.save(os.path.join(sequence_dir, f'seq_{i}_input.npy'), input_seq)
    np.save(os.path.join(sequence_dir, f'seq_{i}_output.npy'), output_seq)