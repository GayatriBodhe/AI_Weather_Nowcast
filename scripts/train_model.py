import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, TimeDistributed  # type: ignore
import numpy as np
import glob

# Load data
input_files = sorted(glob.glob('data/processed/sequences/*_input.npy'))
output_files = sorted(glob.glob('data/processed/sequences/*_output.npy'))

if not input_files or not output_files:
    raise FileNotFoundError("No sequence files found in data/processed/sequences/. Check preprocessing steps.")

X = np.array([np.load(f) for f in input_files])
y = np.array([np.load(f) for f in output_files])

# Verify shapes
print(f"X shape: {X.shape}, y shape: {y.shape}")
if X.shape[1:] != (6, 128, 128, 1) or y.shape[1:] != (12, 128, 128, 1):
    raise ValueError(f"Unexpected shape. Expected (n, 6, 128, 128, 1) and (n, 12, 128, 128, 1), got {X.shape} and {y.shape}")

# Define model
model = Sequential([
    ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(6, 128, 128, 1), padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True),  # Keep sequences for 12 outputs
    BatchNormalization(),
    TimeDistributed(Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same'))
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=4, validation_split=0.2)
model.save('models/nowcast_model.h5')