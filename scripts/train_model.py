#type: ignore 
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Load sequences
input_files = sorted(glob.glob('data/processed/sequences/*_input.npy'))
output_files = sorted(glob.glob('data/processed/sequences/*_output.npy'))

# Load data
X = np.array([np.load(f) for f in input_files])
y = np.array([np.load(f) for f in output_files])

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Model definition
model = Sequential([
    ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(6, 128, 128, 1), padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False),
    BatchNormalization(),
    Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model summary
model.summary()

# Add checkpoint to save after each epoch
os.makedirs('models', exist_ok=True)
checkpoint = ModelCheckpoint('models/nowcast_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
history = model.fit(X, y, epochs=5, batch_size=4, validation_split=0.2, verbose=1, callbacks=[checkpoint])

# Final save (redundant but ensures the last epoch is saved)
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