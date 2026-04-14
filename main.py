from src.data_loader import load_data
from src.preprocess import preprocess_data

from src.model import build_model
from src.train import train_model
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import random

# Fix randomness
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load data
X, y = load_data("data/")

# Preprocess
X = preprocess_data(X)

# Build model
model = build_model()

# Train
history = train_model(model, X, y)

# Save model
model.save("models/model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Validation'])
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], marker='o')
plt.plot(history.history['val_accuracy'], marker='o')

plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend(['Train Accuracy', 'Validation Accuracy'])
plt.grid(True)

plt.savefig("outputs/accuracy.png")
plt.show()