import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# image settings are grouped near the top so beginners can customize them easily.
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10


