import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the trained Mumbai model
model = tf.keras.models.load_model("models/nowcast_model_mumbai.h5")

# Load the last sequence from training data as test input
test_seq = np.load("data/mumbai/processed/sequences/train_x.npy")[-1][np.newaxis, ...]  # Shape: (1, 6, 128, 128, 1)

# Predict 12 frames
predictions = []
current_seq = test_seq.copy()
for _ in range(2):  # 2 iterations of 6 frames = 12 frames total
    pred = model.predict(current_seq, verbose=0)[0]  # Shape: (6, 128, 128, 1)
    predictions.extend([pred[i] for i in range(6)])  # Append each of the 6 predicted frames
    current_seq = np.roll(current_seq, -6, axis=1)  # Shift all 6 frames out
    current_seq[0, -6:] = pred  # Replace the last 6 frames with the prediction

predictions = np.array(predictions)  # Shape: (12, 128, 128, 1)
os.makedirs("data/mumbai/processed/", exist_ok=True)
np.save("data/mumbai/processed/predictions.npy", predictions)

# Save the first prediction with enhanced visualization
plt.imshow(predictions[0, :, :, 0], cmap='gray', vmin=0, vmax=0.5)  # Adjust vmax for contrast
plt.title('Predicted Frame 1 (Mumbai Data)')
plt.colorbar(label='Rain Intensity')
plt.savefig("data/mumbai/processed/prediction.png", dpi=300)  # Higher DPI for clarity
plt.close()
