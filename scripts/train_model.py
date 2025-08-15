import tensorflow as tf
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D #type: ignore
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Load sequences (assuming output_length=1 from create_sequences.py)
input_files = sorted(glob.glob('data/processed/sequences/*_input.npy'))
output_files = sorted(glob.glob('data/processed/sequences/*_output.npy'))

# Load data
X = np.array([np.load(f) for f in input_files])
y = np.array([np.load(f) for f in output_files])

print(f"X shape: {X.shape}, y shape: {y.shape}")  # Should be e.g., (samples, 6, 128, 128, 1), (samples, 1, 128, 128, 1)

# Model definition for single-frame prediction
model = Sequential([
    ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(6, 128, 128, 1), padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False),  # Collapse to single frame
    BatchNormalization(),
    Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')  # Output: (None, 128, 128, 1)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model summary
model.summary()

# Train model
history = model.fit(X, y, epochs=10, batch_size=4, validation_split=0.2, verbose=1)

# Save model
os.makedirs('models', exist_ok=True)
model.save('models/nowcast_model.h5')
print("Model saved to models/nowcast_model.h5")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig('data/processed/training_history.png')
plt.show()