import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models #type: ignore

# Load preprocessed data
x_train = np.load("data/mumbai/processed/sequences/train_x.npy")
y_train = np.load("data/mumbai/processed/sequences/train_y.npy")

# Define model with dropout
model = models.Sequential([
    layers.ConvLSTM2D(64, (3, 3), activation='relu', input_shape=(6, 128, 128, 1), return_sequences=True, padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.ConvLSTM2D(32, (3, 3), activation='relu', return_sequences=True, padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train the model with more epochs
history = model.fit(x_train, y_train, batch_size=4, epochs=20, validation_split=0.2)

# Save the model
os.makedirs("models/", exist_ok=True)
model.save("models/nowcast_model_mumbai_v2.h5")
print("Model saved as models/nowcast_model_mumbai_v2.h5")

# Save training history
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.legend()
plt.savefig("data/mumbai/processed/training_history_v2.png")
plt.close()