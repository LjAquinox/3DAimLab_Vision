import pyautogui
from interception import *
import time
import pyautogui
import random
import numpy as np
import tensorflow as tf  # Assuming you are using TensorFlow
import os
import math
import h5py
from RezizeImg import *
from FindBoundingBoxes import *

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


def move(x,y) :
    move_relative(x, y)

def move_and_click(x,y,delay) :
    move_relative(x, y)
    click(0, 0, delay=0, clicks=1)
    time.sleep(delay)
    click(0, 0, delay=0, clicks=1)
    #click(0, 0, delay=delay)

def restart():
    press2("y",context=context, presses=2, interval=0.5)
    time.sleep(0.01)
    click(0, 0, delay=0)
    time.sleep(0.01)


def main():

    temps = time.time_ns()
    image_number = 0
    restart()
    print("Go")

    while True:

        if time.time_ns() - temps >= 175000000000 :
            restart()
            time.sleep(5)
            restart()
            temps = time.time_ns()


        time.sleep(0.001)

        screenshot = resize_image(np.array(pyautogui.screenshot(region=(300, 200, 1320, 680))), factor=0.5)

        BoundingBoxes = []
        Coords = []

        BoundingBoxes = Find_BB(screenshot,image_number)
        image_number += 1
        Coords = PositionOfBox_RelativeToCenter(BoundingBoxes, factor=0.5)

        for i in Coords :
            mvt = mouse_movement_required_xy(i[0], i[1])
            move_and_click(mvt[0], mvt[1],0.001)
            move(-mvt[0], -mvt[1])


if __name__ == "__main__":
    setSeed(36)
    context = Interception()
    nbFound = 0
    target_color = (255, 120, 55)
    main()
