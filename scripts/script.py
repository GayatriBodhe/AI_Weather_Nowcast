import numpy as np
import matplotlib.pyplot as plt
import os

predictions = np.load("data/mumbai/processed/predictions.npy")
os.makedirs("data/mumbai/processed/", exist_ok=True)
for i in range(12):
    plt.imshow(predictions[i, :, :, 0], cmap='viridis')
    plt.title(f'Predicted Frame {i+1} (Mumbai Data)')
    plt.savefig(f"data/mumbai/processed/prediction_frame_{i+1}.png")
    plt.close()