import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

model = load_model("model.keras")

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (220, 220))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    return np.argmax(pred)

print(predict_image("test.jpg"))
