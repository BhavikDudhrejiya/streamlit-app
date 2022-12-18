from mtcnn import MTCNN
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image as k_image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import asarray

detector = MTCNN()
from deepface.extendedmodels import Age, Gender, Race, Emotion

emotion_model = Emotion.loadModel()
age_model = Age.loadModel()
gender_model = Gender.loadModel()
race_model = Race.loadModel()

race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def detect_face_part(image_input):
    image = cv2.imread(image_input)
    img = cv2.cvtColor(cv2.imread(image_input), cv2.COLOR_BGR2RGB)
    faces_coors = detector.detect_faces(img)[0]['box']
    x,y,w,h = faces_coors
    face_part = image[y:y+h, x:x+w]
    return face_part

def resize_and_convert(img, target_size=(224,224), grayscale=False):
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)
        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)
    img_pixels = k_image.img_to_array(img) 
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255     
    return img_pixels

def detect_emotion(image_input):
    face_part = detect_face_part(image_input)
    resize_face = resize_and_convert(face_part, target_size=(48,48),grayscale=True)
    predict_emotion = emotion_model.predict(resize_face)
    label = emotion_labels[np.argmax(predict_emotion)]
    return label

def detect_age(image_input):
    face_part = detect_face_part(image_input)
    resize_face = resize_and_convert(face_part, target_size=(224,224))
    predict_age = age_model.predict(resize_face)
    apparent_age = Age.findApparentAge(predict_age)
    return int(apparent_age)

def detect_gender(image_input):
    face_part = detect_face_part(image_input)
    resize_face = resize_and_convert(face_part, target_size=(224,224))
    predict_gender = gender_model.predict(resize_face)
    if np.argmax(predict_gender) == 0:
            return 'Woman'
    return 'Man'

def detect_race(image_input):
    face_part = detect_face_part(image_input)
    resize_face = resize_and_convert(face_part, target_size=(224,224))
    predict_race = race_model.predict(resize_face)
    return race_labels[np.argmax(predict_race)]