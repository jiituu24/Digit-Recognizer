import tensorflow as tf 
import numpy as np
import csv

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(25, activation = "relu"),
    Dense(15, activation = "relu"),
    Dense(10, activation = "linear")
])

model.compile(
    loss = SparseCategoricalCrossentropy(from_logits = True),
    optimizer = Adam(0.001)
)
