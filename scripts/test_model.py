import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model('models/nowcast_model.h5')

# Load a test input sequence
current_seq = np.load('data/processed/sequences/test_input.npy')  # Shape (6, 128, 128, 1)

predictions = []
for _ in range(12):
    pred = model.predict(np.expand_dims(current_seq, axis=0))[0]
    predictions.append(pred)
    current_seq = np.roll(current_seq, -1, axis=0)
    current_seq[-1] = pred

predictions = np.stack(predictions, axis=0)
np.save('data/processed/predictions.npy', predictions)

# Visualize first prediction
plt.imshow(predictions[0, :, :, 0], cmap='viridis')
plt.title('Predicted Frame 1')
plt.savefig('data/processed/prediction.png')
plt.show()