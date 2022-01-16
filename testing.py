from main import FaceRecognition
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mtcnn import MTCNN
from PIL import Image
from tensorflow.keras.models import load_model

model_name = "fine_tuning.h5"
model = FaceRecognition.load_saved_model(os.path.join("model", model_name))

# filename = "test.jpg"
# img = plt.imread(filename)
# plt.imshow('image', img)



def model_prediction(image_path, class_names_path):
    class_name = "None Class Name"
    face_array, face = get_detected_face(image_path)

    # model_path = "./model"
    # model = load_model(model_path)

    face_array = face_array.astype('float32')
    input_sample = np.expand_dims(face_array, axis=0)
    result = model.predict(input_sample)
    result = np.argmax(result, axis=1)
    index = result[0]

    classes = np.load(class_names_path, allow_pickle=True).item()
    # print(classes, type(classes), classes.items())
    if type(classes) is dict:
        for k, v in classes.items():
            if k == index:
                class_name = v

    return class_name


def get_detected_face(filename, required_size=(224, 224)):
    img = plt.imread(filename)
    #plt.imshow('image', img)
    #img = cv2.imread(filename)
    #cv2.imshow('image', img)
    detector = MTCNN()
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']  # / todo
    face = img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array, face

#choose img to test prediction
image_path = './testPrediction/test1.jpg'
k = model_prediction(image_path, os.path.join("model", "fine_tuning_class_names.npy"))
print(f"detected class is {k}")


