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

def setSeed(seed) :
    tf.keras.backend.clear_session()
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def key_down2(key: str,context: Interception, delay: Optional[float | int] = None, device: int = 0) -> None:
    keycode = _get_keycode(key)
    stroke = KeyStroke(keycode, KeyState.KEY_DOWN, 0)
    context.send(device,stroke)
    time.sleep(delay or KEY_PRESS_DELAY)


def key_up2(key: str,context: Interception, delay: Optional[float | int] = None, device: int = 0) -> None:
    keycode = _get_keycode(key)
    stroke = KeyStroke(keycode, KeyState.KEY_UP, 0)
    context.send(device,stroke)
    time.sleep(delay or KEY_PRESS_DELAY)

def press2(key: str,context: Interception, presses: int = 1, interval: int | float = 0.1, device: int = 0) -> None:
    for _ in range(presses):
        key_down2(key,context=context,device=device)
        key_up2(key,context=context,device=device)
        if presses > 1:
            time.sleep(interval)

def _get_keycode(key: str) -> int:
    try:
        return KEYBOARD_MAPPING[key]
    except KeyError:
        raise exceptions.UnknownKeyError(key)


def move_and_click(x,y,nbFound) :
    move_relative(x, y)
    time.sleep(0.15)
    condition = pyautogui.pixelMatchesColor(960, 540, target_color, tolerance=5)
    if condition == True:
        pyautogui.screenshot(f'my_screenshot{nbFound}.png', region=(200, 180, 1600, 730))
    time.sleep(0.15)
    click(0, 0, delay=0)
    return condition

def restart():
    press2("y",context=context, presses=2, interval=0.5)
    time.sleep(0.05)
    move_relative(random.randint(-100,100), random.randint(-100,100))
    click(0, 0, delay=0)


class MyNeuralNetwork:
    def __init__(self):
        # Initialize your neural network model here
        self.model = self.build_model()

    def build_model(self):
        # Define and build your neural network architecture here
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(730, 1600, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2)  # Two output units for (x, y) prediction
        ])
        return model

    def train(self, X_train, y_train, num_epochs=5, batch_size=10):

        def customloss(y_true, y_pred) :
            y_pred = float(y_pred)
            y_true = float(y_true)
            return abs((abs(y_pred[0]) - abs(y_true[0])) / 1600 + (abs(y_pred[1]) - abs(y_true[1])) / 900)

        # Compile the model with the mse loss function
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss=customloss)
        # Train the model
        self.model.fit(X_train,y_train, epochs=num_epochs, batch_size=batch_size)

    def predict(self, screenshot):
        # Resize the input screenshot to match the model's input shape
        input_data = np.array(screenshot.resize((1600, 730)))
        input_data = input_data / 255.0  # Normalize pixel values to [0, 1]
        # Perform prediction
        predictions = self.model.predict(np.expand_dims(input_data, axis=0))
        x, y = predictions[0]
        x = int(min(max(int(x*1600),-300),300))
        y = int(min(max(int(y*900),-150),150))
        return x, y



def main(MIN_TRAINING_SAMPLES,SAVE_TRAINING_EVERY,AmountOfRnd):
    temps = time.time_ns()
    nbFound=338
    Value = max(1,int(nbFound / SAVE_TRAINING_EVERY))
    Value2 = max(1,int(nbFound / MIN_TRAINING_SAMPLES))
    TravelDist = (0, 0)
    model = MyNeuralNetwork()
    #print("created model")
    #time.sleep(100)
    X_train = []
    y_train = []
    # Check if there's existing training data and load it
    if os.path.exists("Training_DataX.hdf5"):
        f10 = h5py.File('Training_DataX.hdf5', 'a')
        X_train_arr = f10['X_train']
        f20 = h5py.File('Training_DataY.hdf5', 'a')
        y_train_arr = f20['Y_train']
        del f10
        del f20
        for elem in X_train_arr :
            X_train.append(elem)
        for elem in y_train_arr :
            y_train.append(elem)

    while True:
        if time.time_ns() - temps >= 12000000000 :
            #print(f"Total travel Distance this round {TravelDist}")
            restart()
            temps=time.time_ns()
            TravelDist = (0, 0)

        #print("Taking a screenshot")
        screenshot = pyautogui.screenshot(region=(200, 180, 1600, 730))
        if random.randint(0,100) >= AmountOfRnd :
            x, y = model.predict(screenshot)
        else :
            x, y = (random.randint(-200,200),random.randint(-100,100))
        #print("Screenshot taken")
        cond =move_and_click(x, y,nbFound)
        #print(f"Click x:{x} y:{y}")
        if cond:
            #X_Train.append(screenshot)
            #y_Train.append([x, y])
            #print(f"screenshot shape before append{np.array(screenshot).shape}")
            X_train.append(np.array(screenshot))
            y_train.append([x, y])
            nbFound += 1
            #print(f"X train shape after append{X_train.shape}")
            #print("Found correct color saving it")

        TravelDist = (TravelDist[0] + x, TravelDist[1] + y)
        if abs(TravelDist[0]) >= 300 or abs(TravelDist[1]) >= 150 :
            #print("Hors zone retour a départ")
            move_relative(-int(TravelDist[0]*1.25),-int(TravelDist[1]*1.25))
            TravelDist = (-int(TravelDist[0]*0.25),-int(TravelDist[1]*0.25))
        #time.sleep(5)
        # Save training data periodically
        #print(len(X_train))
        if len(X_train) >= SAVE_TRAINING_EVERY * Value:
            X_train_arr = np.array(X_train)
            y_train_arr = np.array(y_train)
            #print(f"Saving {Value}")
            Value = Value + 1
            f1 = h5py.File("Training_DataX.hdf5", "w")
            #print(f"X train shape before save{X_train_arr.shape}")
            dset1 = f1.create_dataset("X_train", data=X_train_arr)
            f2 = h5py.File("Training_DataY.hdf5", "w")
            dset2 = f2.create_dataset("Y_train", data=y_train_arr)
            del f1
            del f2
            del dset1
            del dset2

        # Train on data periodically
        if len(X_train) >= MIN_TRAINING_SAMPLES * Value2:
            #print(f"Training {Value2}")
            Value2 = Value2 + 1
            X_train_arr = np.array(X_train)  # Convert to NumPy array
            y_train_arr = np.array(y_train)  # Convert to NumPy array
            model.train(X_train_arr, y_train_arr,num_epochs=5)




if __name__ == "__main__":
    setSeed(36)
    MIN_TRAINING_SAMPLES = 5000  # Define the minimum training samples needed
    SAVE_TRAINING_EVERY = 30  # Define how often to save training data
    AmountOfRnd = 50
    context = Interception()
    #context.set_filter(context.is_keyboard, FilterKeyState.FILTER_KEY_DOWN) #To disable keyboard interaction
    screenshot = pyautogui.screenshot()
    nbFound = 0
    target_color = (255, 120, 55)
    main(MIN_TRAINING_SAMPLES,SAVE_TRAINING_EVERY,AmountOfRnd)
