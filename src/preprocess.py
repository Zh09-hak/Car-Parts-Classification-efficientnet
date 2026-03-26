import cv2
import numpy as np
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

def load_data(path, labels):
    X, y = [], []

    for cls in os.listdir(path):
        for file in os.listdir(f"{path}/{cls}"):
            img = cv2.imread(f"{path}/{cls}/{file}")
            img = cv2.resize(img, (220, 220))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(img)

            X.append(img)
            y.append(labels[cls])

    return np.array(X), np.array(y)
