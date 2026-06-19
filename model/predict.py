import time
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

IMAGE_PATH = r"testing\not_fractured\1-rotated1-rotated2-rotated3.jpg"


MODEL_PATH = "bone_fracture_resnet50_model.keras"

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


IMG_SIZE = 224

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("Image path is incorrect!")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

image = np.array(image)

image = preprocess_input(image)

image = np.expand_dims(image, axis=0)

start_time = time.time()

prediction = model.predict(image)

end_time = time.time()

inference_time = end_time - start_time

confidence = float(prediction[0][0])

if confidence < 0.5:
    predicted_class = "Fractured"
    confidence = 1 - confidence
else:
    predicted_class = "Not Fractured"

print("\nPrediction Result")
print("---------------------")
print("Detected Class :", predicted_class)
print("Confidence     :", round(confidence * 100, 2), "%")
print("Inference Time :", round(inference_time, 4), "seconds")