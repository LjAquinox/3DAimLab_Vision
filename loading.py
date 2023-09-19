import pyautogui
from interception import *
import time
import pyautogui
import random
import numpy as np
import tensorflow as tf  # Assuming you are using TensorFlow
import os
import math
import tempfile
import h5py
seed = 1
tf.keras.backend.clear_session()
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

from keras.callbacks import ModelCheckpoint

checkpoint_filepath = './models'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    save_freq='epoch',
)

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print("Invalid device or cannot modify virtual devices once initialized.")
  pass



class MyNeuralNetwork:
    def __init__(self):
        # Initialize your neural network model here
        self.model = self.build_model()

    def build_model(self):
        # Define and build your neural network architecture here
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(730, 1600, 3)),
            tf.keras.layers.MaxPooling2D((3, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2)  # Two output units for (x, y) prediction
        ])
        return model

    def train(self, X_training, Y_training, X_test, Y_test, num_epochs=5, batch_size=5, lr = 0.001):

        def customloss(y_true, y_pred):
            y_pred = float(y_pred)
            y_true = float(y_true)
            return abs((abs(y_pred[0]) - abs(y_true[0])) / 1600 + (abs(y_pred[1]) - abs(y_true[1])) / 900)

        # Compile the model with the mse loss function
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                           loss=customloss)
        # Train the model
        self.model.fit(X_training, Y_training, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test,Y_test), validation_batch_size=batch_size, shuffle=True, callbacks=[model_checkpoint_callback])

    def predict(self, screenshot):
        # Resize the input screenshot to match the model's input shape
        input_data = np.array(screenshot.resize((1600, 730)))
        print(f"screenshot size after resize : {input_data.size}")
        input_data = input_data / 255.0  # Normalize pixel values to [0, 1]

        # Perform prediction
        predictions = self.model.predict(np.expand_dims(input_data, axis=0))
        x, y = predictions[0] #maybe wrong
        x = int(x*1600)
        y = int(y*900)
        return x, y


f = h5py.File('training_dataX.hdf5', 'r')
X_train = f['X_train']
f = h5py.File('training_dataY.hdf5', 'r')
y_train = f['Y_train']
model = MyNeuralNetwork()
print("created model")
model.model.summary()
X_train = np.array(X_train)
y_train = np.array(y_train)



#print(f"x_Train : {X_train}")
print(f"shape X train : {X_train.shape}, shape Y train : {y_train.shape}")
print(f"size of X_train : {X_train.size}, size of X_train[0] : {X_train[0].size}, size of X_train[0][0] : {X_train[0][0].size}, size of X_train[0][0][0] : {X_train[0][0][0].size}")
#print(f"y_Train : {y_train}")


X_training, X_test = X_train[:550, :], X_train[550:650, :]
Y_training, Y_test = y_train[:550, :], y_train[550:650, :]
X_train = None
del X_train
model.train(X_training, Y_training,X_test,Y_test, num_epochs=20, batch_size=2,lr=0.0008)