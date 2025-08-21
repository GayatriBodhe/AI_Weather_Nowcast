import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

# Load the trained Mumbai model
model = tf.keras.models.load_model("models/nowcast_model_mumbai.h5")

# Load all sequences and find one ending at ~16:19 IST (11:19 UTC) on Aug 18
sequences = np.load("data/mumbai/processed/sequences/train_x.npy")
frame_files = sorted(glob.glob("/media/gayatri/Elements/mumbai/raw/*.png"))
target_time_utc = "20250818T1119"  # Approx 11:19 UTC (16:19 IST), adjust based on exact filename
test_seq_idx = next((i for i, f in enumerate(frame_files[-6:]) if target_time_utc in os.path.basename(f)[:13]), -1)
if test_seq_idx == -1:
    print("Sequence for 16:19 IST (11:19 UTC) not found; using last sequence.")
    test_seq = sequences[-1][np.newaxis, ...]
else:
    test_seq = sequences[test_seq_idx][np.newaxis, ...]

# Predict 12 frames (first frame ~16:49 IST)
predictions = []
current_seq = test_seq.copy()
for _ in range(2):  # 2 iterations of 6 frames = 12 frames total
    pred = model.predict(current_seq, verbose=0)[0]  # Shape: (6, 128, 128, 1)
    predictions.extend([pred[i] for i in range(6)])  # Append each of the 6 predicted frames
    current_seq = np.roll(current_seq, -6, axis=1)  # Shift all 6 frames out
    current_seq[0, -6:] = pred  # Replace the last 6 frames with the prediction

predictions = np.array(predictions)  # Shape: (12, 128, 128, 1)
os.makedirs("data/mumbai/processed/", exist_ok=True)
np.save("data/mumbai/processed/predictions_1649.npy", predictions)

# Save the first prediction with enhanced visualization
plt.imshow(predictions[0, :, :, 0], cmap='gray', vmin=0, vmax=0.5)  # Adjust vmax for contrast
plt.title('Predicted Frame at 16:49 IST (Aug 18, 2025)')
plt.colorbar(label='Rain Intensity')
plt.savefig("data/mumbai/processed/prediction_1649.png", dpi=300)  # Higher DPI for clarity
plt.close()